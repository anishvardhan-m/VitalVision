import time

import cv2


class Camera:
    """Manage the VitalVision webcam."""

    def __init__(self, camera_index=0, width=1280, height=720):
        self.camera_index = camera_index

        self.capture = cv2.VideoCapture(camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}."
            )

        # Request the desired resolution.
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )

        # Read the actual resolution provided by the camera.
        self.width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # This is the FPS reported by the camera/backend.
        # We will NOT rely on this value for rPPG processing.
        self.fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        print(
            f"Camera initialized: "
            f"{self.width}x{self.height} @ {self.fps:.2f} FPS"
        )

    def read(self):
        """Read one frame and return it with a timestamp."""

        success, frame = self.capture.read()

        if not success:
            return None, None

        # Use a monotonic clock because it is appropriate
        # for measuring elapsed time between video frames.
        timestamp = time.monotonic()

        return frame, timestamp

    def release(self):
        """Release the webcam."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print("Camera released.")


def main():
    """Run a camera test and measure actual FPS."""

    camera = Camera()

    print("Press Q to quit.")

    frame_count = 0
    start_time = time.monotonic()

    try:
        while True:
            frame, timestamp = camera.read()

            if frame is None:
                print("ERROR: Failed to read camera frame.")
                break

            frame_count += 1

            # Print the first five timestamps so we can verify
            # that frames are being timestamped correctly.
            if frame_count <= 5:
                print(
                    f"Frame {frame_count}: "
                    f"timestamp={timestamp:.6f}"
                )

            elapsed = time.monotonic() - start_time

            # Measure actual frames received every 5 seconds.
            if elapsed >= 5.0:
                measured_fps = frame_count / elapsed

                print(
                    f"Measured FPS: {measured_fps:.2f}"
                )

                frame_count = 0
                start_time = time.monotonic()

            # Display the live camera feed.
            cv2.imshow(
                "VitalVision - Camera Test",
                frame,
            )

            # Press Q while the camera window has focus to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nCamera test interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()