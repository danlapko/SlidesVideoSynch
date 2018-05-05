import json

import cv2

from ocr import get_slide_texts, best_slide, get_image_text, filter_text
from video_processing import video_to_slide_imgs

slides_path = "data/logic_lec9.pdf"
video_path = "data/logic_short2.mp4"

time_period = 1  # sec
max_cntr_area = 0.95  # detected slide rectangle max area
min_cntr_area = 0.3

lang = "rus+eng"
text_len_tresh = 0.1  # min ratio (detected text length)/(slide text length)

if __name__ == "__main__":

    slide_texts = get_slide_texts(slides_path)
    slide_texts = [filter_text(text) for text in slide_texts]
    print("len slides:", len(slide_texts))

    video_imgs = video_to_slide_imgs(video_path, time_period, max_cntr_area, min_cntr_area)
    print("len video_imgs:", len(video_imgs))

    result = dict()

    for tmstmp, img in video_imgs.items():
        cv2.imwrite("tmp/cool{}_.png".format(tmstmp), img)
        img_text = get_image_text(img, lang=lang)
        img_text = filter_text(img_text)
        best_slide_i, ratio = best_slide(img_text, slide_texts, text_len_tresh)
        result[tmstmp] = best_slide_i
        print(tmstmp, ":", "slide=", best_slide_i, "ratio=", ratio, '\t', img_text)

    with open("timestamps.txt", "w") as f:
        f.write(json.dumps(result, ensure_ascii=False))
