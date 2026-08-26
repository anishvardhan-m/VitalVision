import cv2
import numpy as np

from backend.camera import Camera
from backend.face_landmarker import FaceLandmarker
from backend.roi import FaceROI


def overlay_mask(
    frame,
    mask,
    color,
    alpha=0.35,
):
    """Overlay an ROI mask on the camera frame."""

    overlay = frame.copy()

    overlay[mask > 0] = color

    return cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
    )


def draw_roi_contour(
    frame,
    mask,
    color,
):
    """Draw the boundary of an ROI."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        frame,
        contours,
        -1,
        color,
        2,
    )


def main():
    """Run the live ROI visualization test."""

    camera = Camera()
    landmarker = FaceLandmarker()
    roi = FaceROI()

    print("Starting VitalVision ROI test.")
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

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                masks = roi.get_masks(
                    frame,
                    face_landmarks,
                )

                frame = overlay_mask(
                    frame,
                    masks["forehead"],
                    (0, 0, 255),
                )

                frame = overlay_mask(
                    frame,
                    masks["left_cheek"],
                    (0, 255, 0),
                )

                frame = overlay_mask(
                    frame,
                    masks["right_cheek"],
                    (0, 255, 0),
                )

                draw_roi_contour(
                    frame,
                    masks["forehead"],
                    (0, 0, 255),
                )

                draw_roi_contour(
                    frame,
                    masks["left_cheek"],
                    (0, 255, 0),
                )

                draw_roi_contour(
                    frame,
                    masks["right_cheek"],
                    (0, 255, 0),
                )

                cv2.putText(
                    frame,
                    "Forehead",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Cheeks",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                "VitalVision - ROI Test",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nROI test interrupted.")

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()