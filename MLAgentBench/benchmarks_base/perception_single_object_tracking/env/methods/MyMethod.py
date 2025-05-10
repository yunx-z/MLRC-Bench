import numpy as np
from typing import Dict, Any, List
from methods.BaseMethod import BaseMethod

import torch
import torch.nn.functional as F
from libs.siamfc import SiamFC
from libs.backbones import ResNet22W

# Define a simple configuration for SiamFC
class SiamFCConfig:
    def __init__(self, cos_window_val=1.0, padding_val=0.0, exemplar_size=127, instance_size=255, backbone_stride=8):
        self.cos_window = cos_window_val  # Simplified: in practice, a 2D spatial cosine window
        self.padding = padding_val        # Padding factor for template feature cropping
        # These would be used for actual image preprocessing
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.backbone_stride = backbone_stride # Stride of the backbone network (e.g. ResNet22W output stride)


class ObjectTracker():
  """Object tracker class that tracks a given object in a video using SiamFC.

  This model assumes static boxes, given the first bounding box
  which should be tracked in a sequence, for every frame in the
  remaining sequence it will return the same coordinates.

  """

  def __init__(self):
    """Initializes the ObjectTracker class with SiamFC."""
    self.config = SiamFCConfig()
    # Instantiate SiamFC model with ResNet22W backbone
    # Ensure ResNet22W and SiamFC are on the correct device (e.g., .cuda() if using GPU)
    self.siamfc_model = SiamFC(config=self.config, base=ResNet22W())
    self.siamfc_model.eval() # Set the model to evaluation mode

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ObjectTracker: Initialized using device: {self.device}")
    self.siamfc_model.to(self.device)

    self.current_bb_normalized = None # [y_max, x_max, y_min, x_min] normalized
    self.img_height_pixels = None # Will be needed if inputs are not normalized or for precise scaling
    self.img_width_pixels = None  # Will be needed

  def _preprocess_frame(self, frame_numpy: np.ndarray, target_size: int) -> torch.Tensor:
    """
    Placeholder for frame preprocessing.
    Actual implementation would:
    1. Convert NumPy HWC (0-255) to PyTorch CHW Tensor (0-1 or normalized).
    2. Resize/crop to target_size (e.g., 127x127 for template, 255x255 for search).
    3. Apply mean/std normalization if required by the backbone.
    WARNING: Current input frames from evaluation.py are np.zeros, this will not work.
    This function assumes frame_numpy is a single frame (H, W, C) or (C, H, W).
    For SiamFC, input is typically (B, C, H, W).
    """
    if frame_numpy.shape[0] == 1 and frame_numpy.shape[1] == 1 and frame_numpy.shape[2] == 1:
        # This is likely the placeholder np.zeros((1,1,1)) per frame from evaluation.py
        # Create a dummy tensor of the expected shape for the model to run without crashing.
        # THIS WILL NOT PRODUCE MEANINGFUL TRACKING RESULTS.
        print("Warning: Using dummy tensor due to placeholder input frame.")
        dummy_tensor = torch.zeros(3, target_size, target_size)
        return dummy_tensor.unsqueeze(0).to(self.device) # Add batch dim

    # Crude conversion, assuming HWC, RGB, 0-255 input
    # This needs proper implementation (cv2/PIL for resize/crop, normalization)
    if frame_numpy.ndim == 3 and frame_numpy.shape[2] == 3: # HWC
        img_tensor = torch.from_numpy(frame_numpy).permute(2, 0, 1).float() / 255.0
    elif frame_numpy.ndim == 3 and frame_numpy.shape[0] == 3: # CHW
        img_tensor = torch.from_numpy(frame_numpy).float() / 255.0
    else: # Fallback for unexpected shapes, including the (1,1,1) placeholder
        print(f"Warning: Unexpected frame shape {frame_numpy.shape}, creating dummy tensor.")
        img_tensor = torch.zeros(3, target_size, target_size)

    # Simplistic resize (adaptive_avg_pool2d can work as a stand-in for proper resize)
    img_tensor = F.adaptive_avg_pool2d(img_tensor.unsqueeze(0), (target_size, target_size))
    return img_tensor.to(self.device)


  def track_object_in_video(self, frames: np.ndarray, start_info: Dict[str, Any]
                            )-> tuple[np.ndarray, np.ndarray]:
    """Tracks an object in a video using SiamFC.

    Args:
      frames (np.ndarray): Array of frames representing the video.
                           IMPORTANT: Assumed to be (num_frames, H, W, C) or (num_frames, C, H, W)
                           The current evaluation.py provides placeholder (num_frames, 1,1,1) frames.
      start_info (Dict): Dictionary containing the start bounding box and
        frame ID. Expected keys: 'start_bounding_box' (normalized [y_max,x_max,y_min,x_min]),
        'start_id'.

    Returns:
      Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - np.ndarray: Tracked bounding boxes in normalized [y_max,x_max,y_min,x_min] format.
        - np.ndarray: Corresponding frame IDs.
    """
    output_bounding_boxes = []
    output_frame_ids = []

    # Initial bounding box (normalized: [y_max, x_max, y_min, x_min])
    self.current_bb_normalized = np.array(start_info['start_bounding_box'])

    # Placeholder: actual image dimensions would be needed for precise pixel calculations
    # These might come from metadata if available. For now, this limits precision.
    # self.img_height_pixels = frames.shape[1] # if frames are HWC
    # self.img_width_pixels = frames.shape[2]  # if frames are HWC

    # Preprocess the template frame
    template_frame_numpy = frames[start_info['start_id']]
    # IMPORTANT: True SiamFC crops the template patch from this frame using start_bounding_box
    # and resizes it to self.config.exemplar_size.
    # Here, we pass the (potentially full) frame, preprocessed to exemplar_size.
    template_tensor = self._preprocess_frame(template_frame_numpy, self.config.exemplar_size)

    with torch.no_grad():
        self.siamfc_model.update(template_tensor)

    # Add the initial bounding box at the start frame
    output_bounding_boxes.append(self.current_bb_normalized)
    output_frame_ids.append(start_info['start_id'])

    # Iterate through subsequent frames
    for frame_id in range(start_info['start_id'] + 1, frames.shape[0]):
      current_frame_numpy = frames[frame_id]
      # IMPORTANT: True SiamFC crops a search region around the previous prediction
      # from this frame and resizes it to self.config.instance_size.
      # Here, we pass the (potentially full) frame, preprocessed to instance_size.
      search_tensor = self._preprocess_frame(current_frame_numpy, self.config.instance_size)

      with torch.no_grad():
          response = self.siamfc_model.forward(search_tensor) # (1, 1, H_response, W_response)

      # Postprocess response to get new bounding box (simplified)
      response_map = response.squeeze().cpu().numpy() # H_response, W_response
      response_center_y, response_center_x = response_map.shape[0] // 2, response_map.shape[1] // 2

      # Find peak in response map
      peak_y, peak_x = np.unravel_index(np.argmax(response_map), response_map.shape)

      # Displacement in feature map units
      displacement_y_feat = peak_y - response_center_y
      displacement_x_feat = peak_x - response_center_x

      # Convert displacement to (normalized) image coordinates
      # This assumes the search_tensor covers a region of self.config.instance_size pixels
      # and that self.current_bb_normalized was at the center of this region.
      # Displacement in pixels within the instance_size search window:
      displacement_y_pixels = displacement_y_feat * self.config.backbone_stride
      displacement_x_pixels = displacement_x_feat * self.config.backbone_stride

      # Convert pixel displacement to normalized displacement
      # IMPORTANT: This step requires knowledge of the pixel dimensions of the search window (instance_size)
      # or the full original image if the displacement is applied directly to normalized full-image coordinates.
      # Assuming self.config.instance_size corresponds to the search window extent in pixels.
      # A more robust way involves knowing the actual pixel width/height of the original image.
      # For simplicity, let's assume normalized displacement relative to search window size,
      # and that the scale of the bounding box relative to this search window is somewhat implicitly handled.

      # Normalized bounding box: [y_max, x_max, y_min, x_min]
      bb_h_norm = self.current_bb_normalized[0] - self.current_bb_normalized[2]
      bb_w_norm = self.current_bb_normalized[1] - self.current_bb_normalized[3]
      bb_cy_norm = (self.current_bb_normalized[0] + self.current_bb_normalized[2]) / 2.0
      bb_cx_norm = (self.current_bb_normalized[1] + self.current_bb_normalized[3]) / 2.0

      # This is a simplification: assumes the instance_size corresponds to the full normalized range [0,1]
      # which is not accurate. A better way would be to scale displacement_pixels by image_width/height.
      # For now, this is a rough update of normalized center.
      # Let's assume instance_size pixels approximately maps to a certain fraction of the image,
      # or that the features are somewhat scale-invariant in their displacement mapping.
      # A direct conversion:
      # norm_disp_y = displacement_y_pixels / self.img_height_pixels (if available and if displacement is w.r.t full image)
      # For now, scale displacement relative to a fixed search window pixel size (instance_size)
      # This is a common simplification if not doing full coordinate transformations.
      norm_disp_y = displacement_y_pixels / self.config.instance_size # Fraction of search window
      norm_disp_x = displacement_x_pixels / self.config.instance_size # Fraction of search window

      # Update normalized center (this assumes the search window scaling matches normalized coords scaling)
      new_bb_cy_norm = bb_cy_norm + norm_disp_y
      new_bb_cx_norm = bb_cx_norm + norm_disp_x

      new_y_min = new_bb_cy_norm - bb_h_norm / 2.0
      new_y_max = new_bb_cy_norm + bb_h_norm / 2.0
      new_x_min = new_bb_cx_norm - bb_w_norm / 2.0
      new_x_max = new_bb_cx_norm + bb_w_norm / 2.0

      self.current_bb_normalized = np.array([new_y_max, new_x_max, new_y_min, new_x_min])

      # Clip to [0, 1]
      self.current_bb_normalized = np.clip(self.current_bb_normalized, 0.0, 1.0)

      output_bounding_boxes.append(self.current_bb_normalized)
      output_frame_ids.append(frame_id)

    return np.array(output_bounding_boxes), np.array(output_frame_ids)

  # model inference would be inserted here!!
  def track_object_in_frame(self, frame: np.ndarray,
                            prev_bb: List[float]) -> List[float]:
    """
    This method is effectively replaced by the logic within track_object_in_video
    when using SiamFC, as SiamFC processes frame by frame using its internal state.
    Kept for structure, but not directly called by the new track_object_in_video.
    """
    del frame
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
