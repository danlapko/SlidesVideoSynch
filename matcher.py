import json
import os

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from transform import four_point_transform
from video_processing import add_black_border_to_frame, get_contours


def check_frame_has_rectangle(frame, base_area, max_cntr_area=0.95, min_cntr_area=0.3):
    frame = add_black_border_to_frame(frame, bordersize=3)
    contours = get_contours(frame)  # 10 largest contours

    for i_cnt, cnt in enumerate(contours):
        area = cv.contourArea(cnt)

        # checking cnt area
        if area >= base_area * max_cntr_area or area <= base_area * min_cntr_area:
            return False

        # simplifying contour border (reducing the number of cnt points)
        peri = cv.arcLength(cnt, True)
        approx_cnt = cv.approxPolyDP(cnt, 0.01 * peri, True)

        # we are interested only in 4 corners contours
        if 4 <= len(approx_cnt) <= 4:
            # cropping from frame and perspective transformation of contour
            return True
        return False


def get_video_frames(file_path, every):
    cap = cv.VideoCapture(file_path)

    if not cap.isOpened():
        raise FileExistsError("Cannot open file", file_path)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float
    base_cnt = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    base_area = cv.contourArea(base_cnt)

    frames = []
    i_frame = 0
    while True:
        ret, frame = cap.read()
        if cv.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        if i_frame % every == 0:
            # if not check_frame_has_rectangle(frame, base_area):
            #     continue
            tmstmp = int(cap.get(cv.CAP_PROP_POS_MSEC))
            frames.append((tmstmp, frame))

        if i_frame % 5000 == 0:
            print("\tframe", i_frame)
        i_frame += 1

    return frames


def get_pdf_pages(dir):
    names = os.listdir(dir)
    names = sorted(names, key=lambda x: -int(x))
    print("names=", names)
    frames = []
    for name in names:
        frames.append(cv.imread(dir + "/" + name, 0))
    return frames


def match_two_images(img1, img2, n_matches):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches[:n_matches], key=lambda x: x.distance)

    # print([m.distance for m in matches])
    score = sum([mtch.distance for mtch in matches])

    # Draw first 10 matches.
    # img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    # plt.imshow(img3), plt.show()
    return score


def replace_repeated_values_from_list(mylist):
    filtered_list = []

    for i, sublist in enumerate(mylist):

        if 0 < i < len(mylist) - 1:
            sublist_prev = mylist[i - 1]
            sublist_next = mylist[i + 1]

            if sublist_prev[1] == sublist[1] and sublist[1] == sublist_next[1]:
                continue

        filtered_list.append(sublist)
    return filtered_list


def best_slide_ml(frame, slides):
    best_score = 1000_000_000
    i_best_slide = -1
    for i_slide, slide in enumerate(slides):
        score = match_two_images(frame, slide, n_matches)
        if score < tresh_score and score < best_score:
            best_score = score
            i_best_slide = i_slide

    return i_best_slide, best_score


slides_dir = "ir/subdir"
video_pth = "ir/part1.mp4"
output_path = "ir/output.txt"
every_frame = 25 * 50
n_matches = 100
tresh_score = 1000 * n_matches


def main():
    print("read slides ... ")
    slides = get_pdf_pages(slides_dir)
    print("done")

    print("read frames ... ")
    frames = get_video_frames(video_pth, every_frame)
    print("done")

    result = []
    for i_frame, (tmstmp, frame) in enumerate(frames):
        i_best_slide, score = best_slide_ml(frame, slides)
        if i_best_slide > -1:
            result.append([tmstmp, i_best_slide])

        if i_frame % 10 == 0:
            print("\tframe ", i_frame, f"time {tmstmp/1000/60:.4} min, best_slide {i_best_slide}")

    result = replace_repeated_values_from_list(result)
    with open(output_path, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    # pass
    main()

# img1 = cv.imread('base.png', 0)  # trainImage
# img2 = cv.imread('template.png', 0)  # queryImage
# img3 = cv.imread('wrong.png', 0)  # queryImage
# img4 = cv.imread('street.jpg', 0)  # queryImage

# print(match_two_images(img1, img2, n_matches)/n_matches)
# print(match_two_images(img1, img3, n_matches)/n_matches)
# print(match_two_images(img1, img4, n_matches)/n_matches)
