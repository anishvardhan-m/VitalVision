from pathlib import Path

import cv2
import mediapipe as mp


class FaceLandmarker:
    """Manage MediaPipe Face Landmarker."""

    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent

            model_path = (
                base_dir
                / "models"
                / "face_landmarker.task"
            )

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: "
                f"{self.model_path}"
            )

        # Configure the MediaPipe Tasks API.
        #
        # We explicitly use the CPU delegate because
        # MediaPipe 1.0.1 caused a native Metal crash
        # on this Mac. MediaPipe 0.10.35 works correctly.
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        print(
            f"Face Landmarker loaded: "
            f"{self.model_path}"
        )

    def detect(self, frame, timestamp):
        """
        Detect facial landmarks in an OpenCV BGR frame.

        Args:
            frame: OpenCV BGR image.
            timestamp: Frame timestamp in seconds.

        Returns:
            MediaPipe FaceLandmarkerResult.
        """

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # Convert the NumPy array into a MediaPipe Image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # MediaPipe VIDEO mode expects the timestamp
        # in milliseconds.
        timestamp_ms = int(timestamp * 1000)

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return result

    def close(self):
        """Release MediaPipe resources."""

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        print("Face Landmarker closed.")