"""
Video preprocessing utilities

This module contains functions for video preprocessing and frame extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def extract_frames(
    video_path: str, frame_interval: int = 1, max_frames: Optional[int] = None
) -> List[np.ndarray]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        frame_interval: Interval between extracted frames
        max_frames: Maximum number of frames to extract

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_count += 1

    finally:
        cap.release()

    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }

    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()
    return info
