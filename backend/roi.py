import cv2
import numpy as np


class FaceROI:
    """Generate skin-region masks from MediaPipe face landmarks."""

    # ------------------------------------------------------------------
    # Forehead
    # ------------------------------------------------------------------
    #
    # Central forehead region.
    #
    # Designed to stay:
    # - below the hairline
    # - above the eyebrows
    # - away from the eyes
    #
    FOREHEAD = [
        10,
        109,
        67,
        103,
        104,
        105,
        66,
        107,
        9,
        336,
        296,
        334,
        333,
        332,
        297,
        338,
    ]

    # ------------------------------------------------------------------
    # Left cheek
    # ------------------------------------------------------------------
    #
    # Lower cheek region.
    #
    # Avoids:
    # - eyes
    # - eyebrows
    # - mouth
    #
    LEFT_CHEEK = [
        50,
        101,
        118,
        119,
        120,
        100,
        142,
        203,
        205,
        206,
        207,
        187,
    ]

    # ------------------------------------------------------------------
    # Right cheek
    # ------------------------------------------------------------------
    #
    # Mirror of the left cheek region.
    #
    RIGHT_CHEEK = [
        280,
        330,
        347,
        348,
        349,
        329,
        371,
        423,
        425,
        426,
        427,
        411,
    ]

    # ------------------------------------------------------------------
    # Landmark conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_points(
        landmarks,
        indices,
        width,
        height,
    ):
        """
        Convert MediaPipe normalized landmarks
        into OpenCV pixel coordinates.

        MediaPipe coordinates:

            x = 0.0 -> left
            x = 1.0 -> right
            y = 0.0 -> top
            y = 1.0 -> bottom
        """

        points = []

        for index in indices:
            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            # Keep coordinates inside the image.
            x = max(
                0,
                min(width - 1, x),
            )

            y = max(
                0,
                min(height - 1, y),
            )

            points.append([x, y])

        return np.array(
            points,
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Mask creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_mask(
        frame_shape,
        points,
        use_convex_hull=False,
    ):
        """
        Create a filled polygon mask.

        Args:
            frame_shape:
                Shape of the OpenCV frame.

            points:
                Polygon points in pixel coordinates.

            use_convex_hull:
                If True, create the ROI using the convex hull
                of the supplied points.

        Returns:
            Binary uint8 mask.

            Pixels inside the ROI = 255
            Pixels outside the ROI = 0
        """

        height, width = frame_shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        # A polygon requires at least three points.
        if len(points) < 3:
            return mask

        polygon = points

        # --------------------------------------------------------------
        # Convex hull
        # --------------------------------------------------------------
        #
        # The cheek landmark points are not always ordered in a way
        # that produces a clean polygon. A convex hull gives us a
        # stable outer boundary around the cheek landmarks.
        #
        if use_convex_hull:
            polygon = cv2.convexHull(points)

        cv2.fillPoly(
            mask,
            [polygon],
            255,
        )

        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_masks(
        self,
        frame,
        face_landmarks,
    ):
        """
        Create masks for the forehead and both cheeks.

        Args:
            frame:
                OpenCV BGR image.

            face_landmarks:
                One MediaPipe face landmark list.

        Returns:
            Dictionary containing:

                {
                    "forehead": forehead_mask,
                    "left_cheek": left_cheek_mask,
                    "right_cheek": right_cheek_mask,
                }
        """

        height, width = frame.shape[:2]

        # --------------------------------------------------------------
        # Convert MediaPipe landmarks to pixel coordinates
        # --------------------------------------------------------------

        forehead_points = self._landmarks_to_points(
            face_landmarks,
            self.FOREHEAD,
            width,
            height,
        )

        left_cheek_points = self._landmarks_to_points(
            face_landmarks,
            self.LEFT_CHEEK,
            width,
            height,
        )

        right_cheek_points = self._landmarks_to_points(
            face_landmarks,
            self.RIGHT_CHEEK,
            width,
            height,
        )

        # --------------------------------------------------------------
        # Create masks
        # --------------------------------------------------------------

        # Forehead:
        # Keep the original polygon because its landmark ordering
        # already gives us the desired central forehead shape.
        forehead_mask = self._create_mask(
            frame.shape,
            forehead_points,
            use_convex_hull=False,
        )

        # Cheeks:
        # Use convex hull to produce cleaner and more stable regions.
        left_cheek_mask = self._create_mask(
            frame.shape,
            left_cheek_points,
            use_convex_hull=True,
        )

        right_cheek_mask = self._create_mask(
            frame.shape,
            right_cheek_points,
            use_convex_hull=True,
        )

        # --------------------------------------------------------------
        # Return all ROI masks
        # --------------------------------------------------------------

        return {
            "forehead": forehead_mask,
            "left_cheek": left_cheek_mask,
            "right_cheek": right_cheek_mask,
        }