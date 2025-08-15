import os
import cv2
import numpy as np

from utils import stackImages
from mcq_helpers import (
    get_rect_contours,
    get_approx,
    warp_perspective,
    split_boxes,
    draw_answers,
)

# --- Configuration ---
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
NO_QUESTIONS = 5
NO_OPTIONS = 5
IMAGE_PATH = "example.png"
CORRECT_ANSWERS = [3, 2, 1, 2, 0]

# --- Create output folder ---
save_count = 0
folder_count = 0
while os.path.exists(f"Folder {folder_count}"):
    folder_count += 1
os.makedirs(name=f"Folder {folder_count}")

# --- Camera setup ---
use_webcam = True
webcam = cv2.VideoCapture(0)

# --- Main loop ---
while True:
    # Get frame from webcam or image
    if use_webcam:
        is_successful, frame = webcam.read()
    else:
        frame = cv2.imread(filename=IMAGE_PATH)

    frame = cv2.resize(src=frame, dsize=(FRAME_WIDTH, FRAME_HEIGHT))
    contour_frame = frame.copy()
    output_frame2 = frame.copy()

    # Preprocessing
    grayscaled_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    blur_grayscaled_frame = cv2.GaussianBlur(
        src=grayscaled_frame, ksize=(5, 5), sigmaX=1
    )
    edges = cv2.Canny(image=blur_grayscaled_frame, threshold1=100, threshold2=100)

    try:
        # Find rectangular contours
        rect_contours = get_rect_contours(edges)
        biggest_points = get_approx(contour=rect_contours[0])
        second_points = get_approx(contour=rect_contours[1])

        if biggest_points.size != 0 and second_points.size != 0:
            # Draw contours
            cv2.drawContours(
                image=contour_frame,
                contours=biggest_points,
                contourIdx=-1,
                color=(0, 255, 0),
                thickness=20,
            )
            cv2.drawContours(
                image=contour_frame,
                contours=second_points,
                contourIdx=-1,
                color=(255, 0, 0),
                thickness=20,
            )
            # Warp perspective for main sheet and score box
            biggest_warp = warp_perspective(
                image=frame,
                points1=biggest_points,
                points2=[
                    [0, 0],
                    [FRAME_WIDTH, 0],
                    [0, FRAME_HEIGHT],
                    [FRAME_WIDTH, FRAME_HEIGHT],
                ],
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
            second_warp = warp_perspective(
                image=frame,
                points1=second_points,
                points2=[[0, 0], [325, 0], [0, 150], [325, 150]],
                width=325,
                height=150,
            )

            # Prepare copies for marking
            drawn_biggest_warp = biggest_warp.copy()
            blank_biggest_warp = np.zeros_like(biggest_warp)

            # Threshold to binary
            biggest_warp = cv2.cvtColor(src=biggest_warp, code=cv2.COLOR_BGR2GRAY)
            _, thresholded_biggest_warp = cv2.threshold(
                src=biggest_warp, thresh=140, maxval=255, type=cv2.THRESH_BINARY_INV
            )

            # Split into answer boxes
            boxes = split_boxes(thresholded_biggest_warp)

            # Evaluate marked answers
            pixel_values = np.zeros(shape=(NO_QUESTIONS, NO_OPTIONS))
            row_count = 0
            column_count = 0
            for box in boxes:
                nonzero_count = cv2.countNonZero(src=box)
                pixel_values[row_count][column_count] = nonzero_count
                column_count += 1
                if column_count == NO_OPTIONS:
                    row_count += 1
                    column_count = 0

            shaded_positions = []
            for row in pixel_values:
                shaded_position = np.argmax(row)
                shaded_positions.append(shaded_position)

            markings = []
            for x, shaded_position in enumerate(shaded_positions):
                if shaded_positions[x] == CORRECT_ANSWERS[x]:
                    markings.append(1)
                else:
                    markings.append(0)

            score = round((sum(markings) / NO_QUESTIONS) * 100, 1)

            # Draw answers
            drawn_biggest_warp = draw_answers(
                image=drawn_biggest_warp,
                no_questions=NO_QUESTIONS,
                no_options=NO_OPTIONS,
                correct_answers=CORRECT_ANSWERS,
                shaded_positions=shaded_positions,
                markings=markings,
            )
            blank_biggest_warp = draw_answers(
                image=blank_biggest_warp,
                no_questions=NO_QUESTIONS,
                no_options=NO_OPTIONS,
                correct_answers=CORRECT_ANSWERS,
                shaded_positions=shaded_positions,
                markings=markings,
            )

            # Warp markings back to original image
            blank_biggest_warpINV = warp_perspective(
                image=blank_biggest_warp,
                points1=[
                    [0, 0],
                    [FRAME_WIDTH, 0],
                    [0, FRAME_HEIGHT],
                    [FRAME_WIDTH, FRAME_HEIGHT],
                ],
                points2=biggest_points,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
            output_frame1 = cv2.addWeighted(
                src1=frame, alpha=1, src2=blank_biggest_warpINV, beta=1, gamma=2
            )

            # Draw score
            blank_second_warp = np.zeros_like(second_warp)
            cv2.putText(
                img=blank_second_warp,
                text=str(score) + "%",
                org=(30, 100),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=3,
                color=(0, 165, 255),
                thickness=3,
            )

            # Warp score box back
            blank_second_warpINV = warp_perspective(
                image=blank_second_warp,
                points1=[[0, 0], [325, 0], [0, 150], [325, 150]],
                points2=second_points,
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
            output_frame2 = cv2.addWeighted(
                src1=output_frame1,
                alpha=0.9,
                src2=blank_second_warpINV,
                beta=1,
                gamma=0,
            )

        stacked_images = stackImages(
            rows=[
                [frame, grayscaled_frame, blur_grayscaled_frame, edges],
                [
                    contour_frame,
                    biggest_warp,
                    thresholded_biggest_warp,
                    drawn_biggest_warp,
                ],
                [
                    blank_biggest_warp,
                    blank_biggest_warpINV,
                    output_frame1,
                    blank_second_warpINV,
                ],
            ],
            scale=0.3,
        )

    except:
        # Fallback if no contours found
        blank_frame = np.zeros_like(frame)

        stacked_images = stackImages(
            rows=[
                [frame, grayscaled_frame, blur_grayscaled_frame, edges],
                [blank_frame, blank_frame, blank_frame, blank_frame],
                [blank_frame, blank_frame, blank_frame, blank_frame],
            ],
            scale=0.3,
        )

    # Display
    cv2.imshow(winname="output frame2", mat=output_frame2)
    cv2.imshow(winname="stacked images", mat=stacked_images)

    key = cv2.waitKey(delay=1)
    if key == 27:  # ESC to quit
        break

    elif key == ord("s") or key == ord("S"):
        # Save output frame
        cv2.imwrite(
            filename=f"Folder {folder_count}/Marked {save_count}.jpg", img=output_frame2
        )
        cv2.putText(
            img=output_frame2,
            text="SAVED",
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imshow(winname="output frame2", mat=output_frame2)
        cv2.waitKey(delay=500)
        save_count += 1

# Cleanup
webcam.release()
cv2.destroyAllWindows()
