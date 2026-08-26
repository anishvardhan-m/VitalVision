import cv2

from backend.camera import Camera
from backend.face_landmarker import FaceLandmarker
from backend.roi import FaceROI
from backend.roi_features import ROIFeatures


def main():
    """Run the live ROI RGB extraction test."""

    print("Starting VitalVision ROI feature test...")

    camera = Camera()
    landmarker = FaceLandmarker()
    roi = FaceROI()

    print("Starting live RGB extraction.")
    print("Press Q to quit.")

    try:
        while True:
            # ----------------------------------------------------------
            # Read camera frame and timestamp
            # ----------------------------------------------------------

            frame, timestamp = camera.read()

            if frame is None:
                print("ERROR: Failed to read camera frame.")
                break

            # ----------------------------------------------------------
            # Detect facial landmarks
            # ----------------------------------------------------------

            result = landmarker.detect(
                frame,
                timestamp,
            )

            # ----------------------------------------------------------
            # Process the detected face
            # ----------------------------------------------------------

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]

                # Generate forehead and cheek masks.
                masks = roi.get_masks(
                    frame,
                    face_landmarks,
                )

                # Extract mean RGB values from each ROI.
                features = ROIFeatures.extract(
                    frame,
                    masks,
                )

                # ------------------------------------------------------
                # Display RGB values
                # ------------------------------------------------------

                y = 30

                for roi_name, rgb in features.items():

                    if rgb is None:
                        text = (
                            f"{roi_name}: no pixels"
                        )

                    else:
                        red, green, blue = rgb

                        text = (
                            f"{roi_name}: "
                            f"R={red:.1f} "
                            f"G={green:.1f} "
                            f"B={blue:.1f}"
                        )

                    cv2.putText(
                        frame,
                        text,
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    y += 35

            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------------------
            # Display camera
            # ----------------------------------------------------------

            cv2.imshow(
                "VitalVision - ROI RGB Test",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nROI feature test interrupted.")

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()

        print("ROI feature test finished.")


if __name__ == "__main__":
    main()