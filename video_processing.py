import cv2
import numpy as np
from transform import four_point_transform


def add_black_border_to_frame(frame, bordersize=3):
    # adding black frame to detect contours connected to border
    frame = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                               borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return frame


def get_contours(frame):
    # gray scale and Canny transform
    frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(frame_gray, 10, 30)

    # get first 10 contours in area sorted order (from max to min)
    _, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    return contours


def get_frames(cap, every):
    frames = []
    i_frame = 0
    print("reading frames ...")
    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        if i_frame % every == 0:
            tmstmp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            frames.append((tmstmp, frame))
            if i_frame % (every * 10) == 0:
                print("\tframe", i_frame)
        i_frame += 1
    print("reading frames done")

    return frames


def video_to_slide_imgs(file_path, every_frame=1, max_cntr_area=0.95, min_cntr_area=0.3):
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise FileExistsError("Cannot open file", file_path)

    frames = get_frames(cap, every_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    base_cnt = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    base_area = cv2.contourArea(base_cnt)

    # frames_period = time_period * cap.get(cv2.CAP_PROP_FPS)
    print()
    print(f"{len(frames)} frames to filter and crop ...")
    result = []
    for i_frame, (tmstmp, frame) in enumerate(frames):

        frame = add_black_border_to_frame(frame, bordersize=3)
        contours = get_contours(frame)  # 10 largest contours

        for i_cnt, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            # checking cnt area
            if area >= base_area * max_cntr_area or area <= base_area * min_cntr_area:
                continue

            # simplifying contour border (reducing the number of cnt points)
            peri = cv2.arcLength(cnt, True)
            approx_cnt = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            # we are interested only in 4 corners contours
            if 4 <= len(approx_cnt) <= 4:
                # cropping from frame and perspective transformation of contour
                trnsfrmed_img = four_point_transform(frame, approx_cnt)
                result.append((tmstmp, trnsfrmed_img))
                print(f"\taccepted frame {i_frame}")
                break

    print(f"filtering-cropping done")

    return result
