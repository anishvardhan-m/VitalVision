import cv2
import numpy as np


class ROIFeatures:
    """Extract numerical features from face ROI masks."""

    @staticmethod
    def mean_rgb(frame, mask):
        """
        Calculate the mean RGB value of pixels inside an ROI.

        OpenCV stores images as BGR, so the channel order is
        converted to RGB before calculating the mean.

        Args:
            frame:
                OpenCV BGR image.

            mask:
                Binary ROI mask where:
                    255 = pixel belongs to ROI
                    0   = pixel outside ROI

        Returns:
            Tuple containing:

                (red, green, blue)

            Returns None if the ROI contains no pixels.
        """

        # Make sure the mask is valid.
        if mask is None:
            return None

        if frame is None:
            return None

        # Extract pixels where mask == 255.
        pixels = frame[mask > 0]

        if len(pixels) == 0:
            return None

        # OpenCV uses BGR.
        mean_bgr = np.mean(
            pixels,
            axis=0,
        )

        blue = float(mean_bgr[0])
        green = float(mean_bgr[1])
        red = float(mean_bgr[2])

        return red, green, blue

    @staticmethod
    def extract(frame, masks):
        """
        Extract mean RGB values from all face ROIs.

        Args:
            frame:
                OpenCV BGR image.

            masks:
                Dictionary returned by FaceROI.get_masks().

        Returns:
            Dictionary containing RGB values for each ROI.
        """

        features = {}

        for roi_name, mask in masks.items():
            rgb = ROIFeatures.mean_rgb(
                frame,
                mask,
            )

            features[roi_name] = rgb

        return features