import numpy as np
import cv2
from matplotlib import pyplot as plt
from transform import four_point_transform

if __name__ == "__main__":
    cap = cv2.VideoCapture("data/cpp_short2.mp4")

    height, width = 726, 1286

    slide = cv2.imread("data/cpp_lec6/cpp_lec6-09.png")
    slide = cv2.resize(slide, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
    # slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
    slide_gray = cv2.cvtColor(slide.copy(), cv2.COLOR_BGR2GRAY)

    base_cnt = np.array([[0, 0], [0, height], [width, height], [width, 0]])
    base_area = cv2.contourArea(base_cnt)

    base_mask = np.zeros((height, width), np.uint8)
    cv2.imshow("cont", cv2.resize(base_mask, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC))
    cv2.imshow("lec", cv2.resize(base_mask, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC))

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    while True:
        _, frame = cap.read()
        bordersize = 3
        frame = cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(frame_gray, 10, 30)

        _, cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        result_cnt = []

        mask = base_mask.copy()
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area >= base_area * 0.95 or area <= base_area * 0.3:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * peri, True)
            cnts[i] = approx

            if 4 <= len(approx) <= 4:
                result_cnt.append(approx)
                cv2.drawContours(mask, approx, -1, (255,), 1)

                cnt_lol = approx
                cnt_image = four_point_transform(frame, cnt_lol)
                if cv2.waitKey(1) & 0xFF == ord('w'):
                    cv2.imwrite("recognized.png", cnt_image)
                cnt_image = cv2.resize(cnt_image, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("cont", cnt_image)

                # res_matching = cv2.matchTemplate(cnt_image, slide, cv2.TM_CCOEFF_NORMED)
                # threshold = 0.8
                # loc = np.where(res >= threshold)
                # print(res_matching.max())
                break

        cv2.drawContours(frame, result_cnt, -1, (0, 255, 0), 3)

        frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("lec", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
