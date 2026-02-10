import numpy as np
import numpy.typing as npt
from typing import Any
import tempfile
import os
import warnings
import cv2

from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, OpenCVVideoBackend

class NPUOpenCVDynamicVideoBackend(OpenCVVideoBackend):


    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = 32,
        sample_fps: int = 1,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        # logger.info("Using NPUOpenCVDynamicVideoBackend")
        import cv2
        temp_path = None
        backend = cls().get_cv2_video_api()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            temp_path = f.name
        cap = cv2.VideoCapture(temp_path) 
        if not cap.isOpened():
            raise ValueError("Could not open video stream")

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of video frames
        original_fps = float(cap.get(cv2.CAP_PROP_FPS)) # Video fps 
        # The timestamp of the rightmost frame, cannot be used to calculate frame 0.
        total_duration = (total_frames_num - 1) / original_fps

        # `sample_fps` is the FPS parameter passed in for sampling,
        # -1 indicates that sampling can be performed directly without FPS limitation.
        if sample_fps > 0:
            # Num_frames is the maximum number of frames to sample. 
            # If fewer frames are sampled at this sample_fps, the update duration will be longer.
            if num_frames >= int(total_duration * sample_fps) + 1:
                num_frames = int(total_duration * sample_fps) + 1
                # Under the new maximum frame rate, the video duration of the rightmost frame,
                # cannot be calculated for frame 0.
                total_duration = min(total_duration, (num_frames - 1) / sample_fps) 
        elif sample_fps != -1:
            raise ValueError(f"requires dataset fps is -1 or greater than 0 but got {sample_fps}")

        sample_frame_timestamps = np.linspace(0, total_duration, num_frames, dtype=float)
        frames_indices = [min(total_frames_num - 1, round(t * original_fps)) for t in sample_frame_timestamps]

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frames_indices), height, width, 3), dtype=np.uint8)

        i = 0
        for frame_idx in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                i += 1
            else:
                # when get a bad frame,continuous finding a next good frame
                next_idx = frame_idx + 1
                while next_idx < total_frames_num:
                    ret, next_frame = cap.read()
                    if ret:
                        frames[i] = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
                        i += 1
                        break
                    next_idx += 1

        if i != len(frames_indices):
            warnings.warn(
                f"Expected reading {len(frames_indices)} frames, "
                f"but only loaded {i} frames from video.",
                UserWarning,
                stacklevel=2
            )
        if temp_path:
            os.remove(temp_path)

        # Use transformers transformers.video_utils.VideoMetadata format.
        metadata = {
            "total_num_frames": total_frames_num,
            "fps": original_fps,
            "duration": total_duration,
            "video_backend": "opencv_dynamic",
            "frames_indices": frames_indices,
            "do_sample_frames": False,
            "sample_frame_timestamps": sample_frame_timestamps
        }
        return frames, metadata


# registerbackend
VIDEO_LOADER_REGISTRY.register("npu_opencv_dynamic")(NPUOpenCVDynamicVideoBackend)
