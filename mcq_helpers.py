import cv2
import numpy as np


def get_rect_contours(edges):
    """
    Finds rectangular contours in an edge-detected image.
    Filters by area > 500 and having exactly 4 vertices.
    """
    rect_contours = []

    # Find external contours in the edge image
    contours, _ = cv2.findContours(
        image=edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    for contour in contours:
        area = cv2.contourArea(contour=contour)

        if area > 500:  # Filter out very small contours
            perimeter = cv2.arcLength(curve=contour, closed=True)

            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(
                curve=contour, epsilon=0.02 * perimeter, closed=True
            )

            # Only keep rectangular shapes (4 points)
            if len(approx) == 4:
                rect_contours.append(contour)

    # Sort contours by area (largest first)
    rect_contours = sorted(rect_contours, key=cv2.contourArea, reverse=True)

    return rect_contours


def get_approx(contour):
    """
    Approximates a contour to a polygon and returns its vertices.
    """
    perimeter = cv2.arcLength(curve=contour, closed=True)
    approx = cv2.approxPolyDP(curve=contour, epsilon=0.02 * perimeter, closed=True)
    return approx


def reorder(points):
    """
    Reorders 4 points to a consistent order:
    [top-left, top-right, bottom-left, bottom-right].
    """
    new_points = np.zeros_like(points)
    points = np.reshape(a=points, newshape=(4, 2))

    sum = np.sum(a=points, axis=1)  # Sum of coordinates
    difference = np.diff(a=points, axis=1)  # Difference between x and y

    new_points[0] = points[np.argmin(sum)]  # Top-left
    new_points[3] = points[np.argmax(sum)]  # Bottom-right
    new_points[1] = points[np.argmin(difference)]  # Top-right
    new_points[2] = points[np.argmax(difference)]  # Bottom-left

    return new_points


def warp_perspective(image, points1, points2, width, height):
    """
    Applies perspective transformation to warp an image
    from points1 to points2 with given width and height.
    """
    points1 = reorder(points1)
    points2 = reorder(points2)

    points1 = np.float32(points1)
    points2 = np.float32(points2)

    matrix = cv2.getPerspectiveTransform(src=points1, dst=points2)
    warped_image = cv2.warpPerspective(src=image, M=matrix, dsize=(width, height))

    return warped_image


def split_boxes(image):
    """
    Splits a grid image into 25 smaller boxes (5x5).
    Used for extracting MCQ answer regions.
    """
    boxes = []
    rows = np.vsplit(ary=image, indices_or_sections=5)  # Split into 5 rows

    for row in rows:
        columns = np.hsplit(
            ary=row, indices_or_sections=5
        )  # Split each row into 5 columns
        for box in columns:
            boxes.append(box)

    return boxes


def draw_answers(
    image, no_questions, no_options, correct_answers, shaded_positions, markings
):
    """
    Draws the detected answers on the image.
    - Green circle for correct answers
    - Red circle for wrong answers
    - Green mark for correct answer if wrong was selected
    """
    box_width = image.shape[1] / no_options
    box_height = image.shape[0] / no_questions

    for i in range(0, no_questions):
        shaded_position = shaded_positions[i]

        # Coordinates for shaded (selected) answer
        x = round((shaded_position * box_width) + box_width / 2)
        y = round((i * box_height) + box_height / 2)

        if markings[i] == 1:  # Correct answer selected
            color = (0, 255, 0)  # Green
        else:  # Wrong answer selected
            color = (0, 0, 255)  # Red

            # Draw the correct answer in green
            correct_answer = correct_answers[i]
            cv2.circle(
                img=image,
                center=(
                    round((correct_answer * box_width) + box_width / 2),
                    round((i * box_height) + box_height / 2),
                ),
                radius=20,
                color=(0, 255, 0),
                thickness=cv2.FILLED,
            )

        # Draw the selected answer
        cv2.circle(
            img=image, center=(x, y), radius=40, color=color, thickness=cv2.FILLED
        )

    return image
