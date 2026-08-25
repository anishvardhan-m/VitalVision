import cv2

from backend.camera import Camera
from backend.face_landmarker import FaceLandmarker


def draw_landmarks(frame, result):
    """Draw detected facial landmarks on the frame."""

    if not result.face_landmarks:
        return frame

    height, width = frame.shape[:2]

    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Only draw points that fall inside the frame.
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1,
                )

    return frame


def main():
    """Run the live Face Landmarker camera test."""

    camera = Camera()
    landmarker = FaceLandmarker()

    print("Starting live face landmark detection.")
    print("Press Q to quit.")

    try:
        while True:
            frame, timestamp = camera.read()

            if frame is None:
                print("ERROR: Failed to read camera frame.")
                break

            result = landmarker.detect(
                frame,
                timestamp,
            )

            frame = draw_landmarks(
                frame,
                result,
            )

            # Display whether a face was detected.
            if result.face_landmarks:
                status = "Face detected"
            else:
                status = "No face detected"

            cv2.putText(
                frame,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "VitalVision - Face Landmarker",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nFace landmark test interrupted.")

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()