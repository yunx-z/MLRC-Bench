import numpy as np
from typing import Dict, Any, List
from methods.BaseMethod import BaseMethod

class ObjectTracker():
  """Object tracker class that tracks a given object in a video.

  This model assumes static boxes, given the first bounding box
  which should be tracked in a sequence, for every frame in the
  remaining sequence it will return the same coordinates.

  """

  def __init__(self):
    """Initializes the ObjectTracker class."""
    pass

  def track_object_in_video(self, frames: np.ndarray, start_info: Dict[str, Any]
                            )-> Dict[str, Any]:
    """Tracks an object in a video.

    Tracks an object given a sequence of frames and initial information about
    the coordinates and frame ID of the object.

    Args:
      frames (np.ndarray): Array of frames representing the video.
      start_info (Dict): Dictionary containing the start bounding box and
        frame ID. Expected keys: 'start_bounding_box', 'start_id'.

    Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - np.ndarray: Tracked bounding boxes.
        - np.ndarray: Corresponding frame IDs.
    """
    # initially take starting bounding box for tracking
    prev_bb = start_info['start_bounding_box']
    output_bounding_boxes = []
    output_frame_ids = []

    for frame_id in range(start_info['start_id'], frames.shape[0]):
      frame = frames[frame_id]
      # here is where the per frame tracking is done by the model
      # we just return the starting coords in this dummy baseline
      bb = self.track_object_in_frame(frame, prev_bb)
      output_bounding_boxes.append(bb)
      output_frame_ids.append(frame_id)

    output_bounding_boxes = np.stack(output_bounding_boxes, axis=0)
    output_frame_ids = np.array(output_frame_ids)
    return output_bounding_boxes, output_frame_ids

  # model inference would be inserted here!!
  def track_object_in_frame(self, frame: np.ndarray,
                            prev_bb: List[float]) -> List[float]:
    """Tracks an object in a single frame.

    Tracks an object in a single frame based on the previous bounding box
    coordinates. Placeholder function that just returns the coords it is given,
    assumes a static object.

    Args:
      frame (np.ndarray): The current frame.
      prev_bb(List): Previous bounding box coordinates. (y2,x2,y1,x1)

    Returns:
      List: The tracked bounding box coordinates in the current frame.
    """
    del frame  # unused
    return prev_bb


class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        # Instantiate the tracker
        self.tracker = ObjectTracker()

    def run(self, input_args: Dict[str, Any]):
        """
        Runs the object tracking algorithm.

        Args:
            input_args (Dict[str, Any]): A dictionary containing the input arguments.
                                         Expected keys: 'frames' (np.ndarray),
                                         'start_info' (Dict).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Results from the tracker.
        """
        frames = input_args.get('frames')
        start_info = input_args.get('start_info')

        if frames is None or start_info is None:
            raise ValueError("Missing 'frames' or 'start_info' in input_args")

        # Call the tracker's method
        results = self.tracker.track_object_in_video(frames=frames, start_info=start_info)

        # Return the results
        return results
