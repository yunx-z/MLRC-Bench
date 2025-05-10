from constants import * # the constants will be replaced with held-out test data/models during test phase
import os
from typing import Tuple, List, Dict, Any
import json
import numpy as np
import cv2 # For video processing
from tqdm import tqdm # For progress bar

from vot.region import calculate_overlaps as calculate_region_overlaps
from vot.region import Polygon, Rectangle, Special

# define or import any evaluation util functions here
def load_db_json(db_file: str) -> Dict[str, Any]:
  """Loads a JSON file as a dictionary.

  Args:
    db_file (str): Path to the JSON file.

  Returns:
    Dict: Loaded JSON data as a dictionary.

  Raises:
    FileNotFoundError: If the specified file doesn't exist.
    TypeError: If the JSON file is not formatted as a dictionary.
  """
  if not os.path.isfile(db_file):
    raise FileNotFoundError(f'No such file: {db_file}')

  with open(db_file, 'r') as f:
    db_file_dict = json.load(f)
    if not isinstance(db_file_dict, dict):
      raise TypeError('JSON file is not formatted as a dictionary.')
    return db_file_dict

def get_start_frame(track_arr: List[List[float]]) -> int:
  """Returns index of the first non-zero element in a track array.

  Args:
    track_arr (list): one hot vector correspoinding to annotations,
      showing which index to start tracking .

  Returns:
    int: Index of the first non-zero element in the track array.

  Raises:
    ValueError: Raises error if the length of the array is 0
      or if there is no one-hot value.
  """
  if not track_arr or np.count_nonzero(track_arr) == 0:
    raise ValueError('Track is empty or has no non-zero elements')
  return np.nonzero(track_arr)[0][0]

def get_start_info(track: Dict[str, Any]) -> Dict[str, Any]:
  """Retrieve information about the start frame of a track.
  Handles cases for dev (with 'initial_tracking_box') and test (without).

  Args:
    track (Dict): A dictionary containing information about the track.
                  Expected keys: 'frame_ids', 'bounding_boxes'.
                  Optional key: 'initial_tracking_box' for dev phase.

  Returns:
    Dict[str: Any]: A dictionary with the following keys:
      'start_id': The frame ID of the start frame.
      'start_bounding_box': The bounding box coordinates of the start frame.
      'start_idx': The index of the start frame in the 'bounding_boxes' list.

  Raises:
    ValueError: If necessary keys are missing or track arrays are empty.
  """
  if 'initial_tracking_box' in track and track['initial_tracking_box'] and np.count_nonzero(track['initial_tracking_box']) > 0:
    # Dev phase or phase with explicit initial_tracking_box
    try:
        track_start_idx_in_initial_array = get_start_frame(track['initial_tracking_box'])
        # This track_start_idx_in_initial_array is the index within the 'frame_ids' and 'bounding_boxes' arrays
        # that corresponds to the start of tracking.
        track_start_idx = track_start_idx_in_initial_array
        if track_start_idx >= len(track['frame_ids']) or track_start_idx >= len(track['bounding_boxes']):
            raise ValueError(f"Start index {track_start_idx} from 'initial_tracking_box' is out of bounds "
                             f"for frame_ids (len: {len(track['frame_ids'])}) or "
                             f"bounding_boxes (len: {len(track['bounding_boxes'])}).")
        track_start_id = track['frame_ids'][track_start_idx]
        track_start_bb = track['bounding_boxes'][track_start_idx]
    except ValueError as e:
        print(f"Could not determine start from 'initial_tracking_box': {e}. Falling back to first frame.")
        if not track.get('frame_ids') or not track.get('bounding_boxes'):
            raise ValueError("Track has no frame_ids or bounding_boxes to infer start info after 'initial_tracking_box' failed.")
        track_start_idx = 0
        track_start_id = track['frame_ids'][0]
        track_start_bb = track['bounding_boxes'][0]
  else:
    # Test phase or if initial_tracking_box is missing, empty, or all zeros.
    # Assume the first frame and bbox in the provided sequence is the start.
    if not track.get('frame_ids') or not track.get('bounding_boxes'):
      raise ValueError("Track has no frame_ids or bounding_boxes to infer start info.")
    track_start_idx = 0 # Index of the start frame in the 'frame_ids' and 'bounding_boxes' lists
    track_start_id = track['frame_ids'][0]
    track_start_bb = track['bounding_boxes'][0]

  return {'start_id': track_start_id,
          'start_bounding_box': track_start_bb,
          'start_idx': track_start_idx}

def filter_pred_boxes(pred_bb: np.ndarray, pred_fid: np.ndarray,
                      gt_fid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Filter bounding boxes and frame IDs based on ground truth frame IDs.

  Args:
    pred_bb (np.ndarray): Array of predicted bounding boxes.
    pred_fid (np.ndarray): Array of frame IDs for predicted bounding boxes.
    gt_fid (np.ndarray): Array of frame IDs for ground truth bounding boxes.

  Returns:
    Tuple[np.ndarray, np.ndarray]: Filtered predicted bounding boxes and
    	their corresponding frame IDs.
  """
  pred_idx = np.isin(pred_fid, gt_fid).nonzero()[0]
  filter_pred_bb = pred_bb[pred_idx]
  filter_pred_fid = pred_fid[pred_idx]
  return filter_pred_bb, filter_pred_fid

def bbox2region(bbox: np.array) -> Rectangle:
  """Convert bbox to Rectangle or Polygon Class object.

  Args:
    bbox (ndarray): the format of rectangle bbox is (x1, y1, w, h);
      the format of polygon is (x1, y1, x2, y2, ...).

  Returns:
    Rectangle or Polygon Class object.

  Raises:
  	NotImplementedError: Returns error if unexpected number of coordinates in
      shape.
  """

  if len(bbox) == 1:
    return Special(bbox[0])
  elif len(bbox) == 4:
    return Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
  elif len(bbox) % 2 == 0 and len(bbox) > 4:
    return Polygon([(x_, y_) for x_, y_ in zip(bbox[::2], bbox[1::2])])
  else:
    raise NotImplementedError(
        f'The length of bbox is {len(bbox)}, which is not supported')

def trajectory2region(trajectory: List) -> List:
  """Convert bbox trajectory to Rectangle or Polygon Class object trajectory.

  Args:
    trajectory (list[ndarray]): The outer list contains bbox of
      each frame in a video. The bbox is a ndarray.

  Returns:
    List: contains the Region Class object of each frame in a
      trajectory.
  """
  traj_region = []
  for bbox in trajectory:
    traj_region.append(bbox2region(bbox))
  return traj_region

def calc_accuracy(gt_trajectory: List, pred_trajectory: List) -> float:
  """Calculate accuracy over the sequence.

  Args:
    gt_trajectory (list[list]): list of bboxes
    pred_trajectory (list[ndarray]): The outer list contains the
      tracking results of each frame in one video. The ndarray has two cases:
      - bbox: denotes the normal tracking box in [x1, y1, w, h]
      	format.
      - special tracking state: [0] denotes the unknown state,
      	namely the skipping frame after failure, [1] denotes the
        initialized state, and [2] denotes the failed state.

  Returns:
    Float: accuracy over the sequence.
  """
  pred_traj_region = trajectory2region(pred_trajectory)
  gt_traj_region = trajectory2region(gt_trajectory)
  overlaps = np.array(calculate_region_overlaps(pred_traj_region,
                                                gt_traj_region))
  mask = np.ones(len(overlaps), dtype=bool)
  return np.mean(overlaps[mask]) if any(mask) else 0.

def run_iou(boxes: Dict[str, Any], db_dict: Dict[str, Any], filter=True) -> Dict[str, Any]:
  """Calculate IoU per track and per video.

  Calculate Intersection over Union (IoU) for predicted and ground truth
    bounding boxes for all tracks in the provided outputs.

  Args:
    boxes (Dict): Dict containing predicted and label bounding boxes for each
      video. Boxes must be in format [x1,y1,x2,y2].
    db_dict (Dict): Dict containing annotations.

  Returns:
    Dict: A dictionary with video IDs as keys and
      a list of IoU scores as values.
  """
  all_vot_iou = {}
  for vid_id, pred_tracks in boxes.items():
    gt_tracks = db_dict[vid_id]['object_tracking']

    video_iou = {}
    for pred_track in pred_tracks:
      gt_track = gt_tracks[pred_track['id']]
      # check track IDs
      assert pred_track['id'] == gt_track['id']

      start_info = get_start_info(gt_track)
      start_idx = start_info['start_idx']
      # get bounding boxes from frame ID were tracking is supposed to start +1
      gt_bb = np.array(gt_track['bounding_boxes'])[start_idx+1:]
      gt_fid = gt_track['frame_ids'][start_idx+1:]
      # weird case where only one box is labelled
      if not gt_fid:
        continue

      pred_bb = np.array(pred_track['bounding_boxes'])
      pred_fid = np.array(pred_track['frame_ids'])
      # filter predicted trajectory for frame IDs where we have annotations
      if filter:
        pred_bb, pred_fid = filter_pred_boxes(pred_bb, pred_fid, gt_fid)

      # check for missing frame IDs in prediction
      missing_idx = np.where(np.isin(gt_fid, pred_fid, invert=True))[0]
      if missing_idx.size != 0:
        raise ValueError(f'Missing IDs from object trajectory: {missing_idx}')
      if len(gt_bb) != len(pred_bb):
        raise ValueError('Missing boxes in predictions')

      #  convert y2,x2,y1,x1 [0,1] to x1,y1,w,h in pixel space
      [height, width] = db_dict[vid_id]['metadata']['resolution']
      pred_w = pred_bb[:, 2] - pred_bb[:, 0]
      pred_h = pred_bb[:, 3] - pred_bb[:, 1]
      pred_bb = np.stack([pred_bb[:, 0]*width, pred_bb[:, 1]*height,
                          pred_w*width, pred_h*height], axis=1)
      gt_w = gt_bb[:, 2] - gt_bb[:, 0]
      gt_h = gt_bb[:, 3] - gt_bb[:, 1]
      gt_bb = np.stack([gt_bb[:, 0]*width, gt_bb[:, 1]*height,
                        gt_w*width, gt_h*height], axis=1)

      # compute IoU per track
      iou = calc_accuracy(gt_bb, pred_bb)
      video_iou[pred_track['id']] = iou

    all_vot_iou[vid_id] = video_iou

  return all_vot_iou

def summarise_results(labels: Dict[str, Any], results: Dict[str, Any]):
  """Summarise the results according to camera movement.

  Summarise the results of a dataset by calculating average IoU scores
  across all videos, videos with a static camera and videos with a moving
  camera.

  Args:
    labels (Dict): A dictionary containing metadata and
      information about the dataset.
    results (Dict): A dictionary containing IoU scores
      for each video in the dataset.
  """
  all_ious = []
  # aggregate performance based on camera motion for analysis
  static_ious = []
  moving_ious = []

  for vid, iou_dict in results.items():
    ious = list(iou_dict.values())
    if not ious:
      continue

    all_ious.append(np.mean(ious))

    if labels[vid]['metadata']['is_camera_moving']:
      moving_ious.append(np.mean(ious))
    else:
      static_ious.append(np.mean(ious))

  if all_ious:
    print(f"""Average IoU across all videos in dataset:
          {np.array(all_ious).mean():.3f}""")

  if static_ious:
    print(f"""Average IoU across static camera videos in dataset:
          {np.array(static_ious).mean():.3f}""")

  if moving_ious:
    print(f"""Average IoU across moving camera videos in dataset:
          {np.array(moving_ious).mean():.3f}""")

# Placeholder for video frame loading function
def get_video_frames(video_data_item: Dict[str, Any], video_folder_path: str) -> np.ndarray:
    """
    Loads all frames from a video file.

    Args:
        video_data_item (Dict[str, Any]): Dictionary containing metadata for the video,
                                       expected to have ['metadata']['video_id'] (or similar for filename)
                                       and ['metadata']['num_frames'].
        video_folder_path (str): Path to the folder containing the video files.

    Returns:
        np.ndarray: A numpy array of video frames (num_frames, height, width, channels).
                    Returns a placeholder if video loading fails or is not fully implemented.
    """
    metadata = video_data_item.get('metadata', {})
    video_id = metadata.get('video_id') # Or other key like 'filename'
    num_frames = metadata.get('num_frames', 0)
    resolution = metadata.get('resolution', (0,0)) # (height, width)

    if not video_id:
        print("Warning: video_id not found in metadata.")
        # Fallback to placeholder if video_id is missing
        return np.zeros((num_frames if num_frames > 0 else 1, resolution[0] if resolution[0]>0 else 1, resolution[1] if resolution[1]>0 else 1, 3), dtype=np.uint8)

    # Assuming video files are named like '{video_id}.mp4' or similar. Adjust as needed.
    # You might need to check for different extensions or a specific filename field in metadata.
    potential_extensions = ['.mp4', '.avi', '.mov', '.mkv'] # Add other common video extensions
    video_path = None
    for ext in potential_extensions:
        test_path = os.path.join(video_folder_path, f"{video_id}{ext}")
        if os.path.exists(test_path):
            video_path = test_path
            break

    if video_path is None: # If video_id itself might be a full filename (e.g. "myvideo.mp4")
        test_path = os.path.join(video_folder_path, video_id)
        if os.path.exists(test_path):
            video_path = test_path

    # Print which video is being attempted
    if video_path and os.path.exists(video_path):
        print(f"Loading frames for video: {video_id} from {video_path}")
    elif video_id:
        print(f"Attempting to find video for ID: {video_id} in {video_folder_path}")

    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video file not found for ID {video_id} in {video_folder_path} (tried extensions: {potential_extensions}, and direct ID as filename).")
        # Fallback to placeholder if video file is missing
        return np.zeros((num_frames if num_frames > 0 else 1, resolution[0] if resolution[0]>0 else 1, resolution[1] if resolution[1]>0 else 1, 3), dtype=np.uint8)

    frames_list = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            # Fallback to placeholder
            return np.zeros((num_frames if num_frames > 0 else 1, resolution[0] if resolution[0]>0 else 1, resolution[1] if resolution[1]>0 else 1, 3), dtype=np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # By default, OpenCV reads frames in BGR format.
            # If RGB is expected by the model, uncomment the line below:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # If frames need to be a specific resolution and differ from video's native resolution:
            # target_height, target_width = resolution
            # if frame.shape[0] != target_height or frame.shape[1] != target_width:
            #     if target_height > 0 and target_width > 0:
            #         frame = cv2.resize(frame, (target_width, target_height))
            #     else:
            #         print(f"Warning: Cannot resize frame for {video_id} due to invalid target resolution {resolution}")

            frames_list.append(frame)
        cap.release()

        if not frames_list: # No frames read
             print(f"Warning: No frames read from {video_path}")
             return np.zeros((num_frames if num_frames > 0 else 1, resolution[0] if resolution[0]>0 else 1, resolution[1] if resolution[1]>0 else 1, 3), dtype=np.uint8)

        loaded_frames_array = np.array(frames_list)

        # Optional: Verify against num_frames from metadata
        # if num_frames > 0 and loaded_frames_array.shape[0] != num_frames:
        #     print(f"Warning: Mismatch in frame count for {video_id}. Metadata: {num_frames}, Actual: {loaded_frames_array.shape[0]}. Adjusting to metadata count if possible or using actual.")
            # Decide on strategy: pad, truncate, or use actual. For now, using actual.
            # If strict adherence to metadata num_frames is needed:
            # if loaded_frames_array.shape[0] > num_frames:
            #    loaded_frames_array = loaded_frames_array[:num_frames]
            # elif loaded_frames_array.shape[0] < num_frames:
            #    padding_shape = (num_frames - loaded_frames_array.shape[0],) + loaded_frames_array.shape[1:]
            #    padding = np.zeros(padding_shape, dtype=loaded_frames_array.dtype) # Or pad with last frame
            #    loaded_frames_array = np.concatenate((loaded_frames_array, padding), axis=0)


        return loaded_frames_array

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        # Fallback to placeholder on error
        return np.zeros((num_frames if num_frames > 0 else 1, resolution[0] if resolution[0]>0 else 1, resolution[1] if resolution[1]>0 else 1, 3), dtype=np.uint8)

class PerceptionDataset():
  """Dataset class to store video items from dataset.

  Attributes:
    video_folder_path: Path to the folder containing the videos.
    task: Task type for annotations.
    split: Dataset split to load.
  	task_db: List containing annotations for dataset according to
  		split and task availability.
  """

  def __init__(self, db_path: Dict[str, Any], video_folder_path: str,
               task: str, split: str) -> None:
    """Initializes the PerceptionDataset class.

    Args:
      db_path (str): Path to the annotation file.
      video_folder_path (str): Path to the folder containing the videos.
      task (str): Task type for annotations.
      split (str): Dataset split to load.
    """
    self.video_folder_path = video_folder_path
    self.task = task
    self.split = split
    self.task_db = self.load_dataset(db_path)

  def load_dataset(self, db_path: str) -> List:
    """Loads the dataset from the annotation file and processes.

    Dict is processed according to split and task.

    Args:
      db_path (str): Path to the annotation file.

    Returns:
      List: List of database items containing annotations.
    """
    db_dict = load_db_json(db_path)
    db_list = []
    for _, val in db_dict.items():
      if val['metadata']['split'] == self.split:
        if val[self.task]:  # If video has annotations for this task
          db_list.append(val)

    return db_list

  def __len__(self) -> int:
    """Returns the total number of videos in the dataset.

    Returns:
      int: Total number of videos.
    """
    return len(self.task_db)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """Returns the video and annotations for a given index.

    Args:
      idx (int): Index of the video.

    Returns:
      Dict: Dictionary containing the video frames, metadata, annotations.
    """
    data_item = self.task_db[idx]
    annot = data_item[self.task]

    metadata = data_item['metadata']

    # Load actual video frames using the new function
    vid_frames = get_video_frames(data_item, self.video_folder_path)

    # Ensure the number of loaded frames matches metadata if critical, or handle discrepancies
    # This check is more for information; the model will receive what's loaded.
    # The 'num_frames' in metadata might be used by other parts of the code, so consistency can be important.
    expected_num_frames = metadata.get('num_frames')
    if expected_num_frames is not None and expected_num_frames > 0 and vid_frames.shape[0] != expected_num_frames:
        print(f"Warning: Loaded frames ({vid_frames.shape[0]}) for video {metadata.get('video_id', 'Unknown')} "
              f"does not match num_frames in metadata ({expected_num_frames}). "
              f"The model will use the actual number of loaded frames: {vid_frames.shape[0]}.")
        # If you need to strictly enforce the metadata's num_frames, you might need to:
        # 1. Truncate `vid_frames` if `vid_frames.shape[0] > expected_num_frames`.
        # 2. Pad `vid_frames` if `vid_frames.shape[0] < expected_num_frames`.
        # For example, to truncate:
        # if vid_frames.shape[0] > expected_num_frames:
        #     vid_frames = vid_frames[:expected_num_frames]
        # Padding is more complex as you need to decide what to pad with (e.g., black frames, last frame).
        # The current implementation will proceed with the actual number of frames loaded.

    return {'metadata': metadata,
            self.task: annot,
            'frames': vid_frames}

def evaluate_model(Method, phase):
    MODEL_SAVE_FILENAME = "object_tracker_model.pth"
    MODEL_SAVE_DIR = './output/saved_models/' # Centralized directory for models

    # Ensure the save directory exists
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FILENAME)

    # 1. load test input data from dataset_filepath
    if phase == "dev":
        label_path = f'./data/{phase}/object_tracking_valid_subset.json'
        video_folder_path = f'./data/{phase}/videos/'
        split_value = "valid"
    else: # Assuming this branch is for "test" or other non-dev phases
        label_path = f'./data/{phase}/object_tracking_{phase}_subset.json'
        # Specific path for test phase videos
        if phase == "test":
            video_folder_path = f'./data/{phase}/test_release/videos/'
        else:
            # Fallback for any other potential phases
            video_folder_path = f'./data/{phase}/videos/'
        split_value = phase

    cfg = {'video_folder_path': video_folder_path,
       'task': 'object_tracking',
       'split': split_value}

    filter_outputs = True # This seems to be a general setting

    # init dataset
    tracking_dataset = PerceptionDataset(label_path, **cfg)

    # init tracking model (Method is already an instance)
    method_instance = Method

    if phase == "test":
        if hasattr(method_instance, 'load_model'):
            print(f"Test phase: Attempting to load model from {model_save_path}")
            method_instance.load_model(model_save_path)
        else:
            print(f"Warning: Method instance {type(method_instance).__name__} does not have 'load_model' capability.")

    # 2. apply the method / model on the whole dev / test data
    results = {}
    print(f"Processing {len(tracking_dataset)} videos for phase: {phase}")
    for video_item in tqdm(tracking_dataset, desc=f"Processing videos for phase {phase}", unit="video"):
        video_id = video_item['metadata']['video_id']
        video_pred_tracks = []

        if not video_item.get('object_tracking'):
            print(f"Warning: No 'object_tracking' data for video_id {video_id}. Skipping.")
            results[video_id] = [] # Ensure video_id is in results
            continue

        for gt_track in video_item['object_tracking']:
            try:
                start_info = get_start_info(gt_track)
            except ValueError as e:
                print(f"Skipping track in video {video_id} due to error in get_start_info: {e}")
                continue # Skip this track if start_info cannot be determined

            input_args = {
                'frames': video_item['frames'],
                'start_info': start_info
            }
            pred_bounding_boxes, pred_frame_ids = method_instance.run(input_args)

            if filter_outputs:
                # Ensure gt_track['frame_ids'] is available for filtering
                gt_frame_ids_for_filtering = np.array(gt_track.get('frame_ids', []))
                if gt_frame_ids_for_filtering.size == 0:
                     print(f"Warning: gt_track['frame_ids'] is empty for video {video_id}, track {gt_track.get('id')}. Cannot filter predictions.")
                     # If gt_frame_ids is empty, filtering might not make sense or predictions should be taken as is.
                     # For now, if gt_frame_ids is empty, we won't filter.
                else:
                    pred_bounding_boxes, pred_frame_ids = (
                        filter_pred_boxes(pred_bounding_boxes, pred_frame_ids, gt_frame_ids_for_filtering)
                    )


            pred_track = {}
            pred_track['bounding_boxes'] = pred_bounding_boxes.tolist()
            pred_track['frame_ids'] = pred_frame_ids.tolist()
            pred_track['id'] = gt_track['id']
            video_pred_tracks.append(pred_track)

        results[video_id] = video_pred_tracks

    # 3. save the results to a file under `./output` (this is for predictions)
    output_dir = './output' # For results.json
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, f'results_{phase}.json')
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filepath}")

    # Save the model if in "dev" phase, after all processing for this phase is done.
    if phase == "dev":
        if hasattr(method_instance, 'save_model'):
            print(f"Dev phase: Attempting to save model to {model_save_path}")
            method_instance.save_model(model_save_path)
        else:
            print(f"Warning: Method instance {type(method_instance).__name__} does not have 'save_model' capability.")


def get_score(Method, phase):
    # 1. load results from `./output`
    output_dir = './output'
    results_filepath = os.path.join(output_dir, f'results_{phase}.json')
    results = load_db_json(results_filepath)

    if phase == "dev":
        label_path = f'./data/{phase}/object_tracking_valid_subset.json'
    else:
        label_path = f'./data/{phase}/object_tracking_{phase}_subset.json'
    label_dict = load_db_json(label_path)

    # when true the bounding boxes and frame IDs will be filtered to demonstrate
    # how to submit for the subset of annotated frames which will be used for
    # evaluation
    filter_outputs = True

    # 2. calculate evaluation metric (either locally or use Kaggle API to submit to the leaderboard)
    iou_results = run_iou(results, label_dict, filter=filter_outputs)
    # 3. (optional) save sample-level evaluation scores to a file (this may not be possible with Kaggle API evaluation)
    # 4. use `print()` function to print a message informing the evaluation score, which will be visible to LLM agents.
    summarise_results(label_dict, iou_results)
    # 5. return the final score (a single number, the higher the better)
    all_ious = []
    for vid, iou_dict in iou_results.items():
        # Get IoU scores for all tracks in the current video
        ious = list(iou_dict.values())
        # Skip if there are no IoU scores for this video
        if not ious:
            continue
        # Calculate the mean IoU for the current video and add it to the list
        all_ious.append(np.mean(ious))

    # Calculate the final score as the mean of mean IoUs across all videos
    # If all_ious is empty (no valid videos/tracks), return 0.0
    final_score = np.array(all_ious).mean() if all_ious else 0.0

    return final_score

