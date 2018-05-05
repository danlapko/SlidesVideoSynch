import argparse
import json
from collections import OrderedDict

import cv2

from ocr import get_slide_texts, best_slide, get_image_text, filter_text
from video_processing import video_to_slide_imgs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slides_path", help="in pdf format", type=str, default="tests/example.pdf")
    parser.add_argument("-v", "--video_path", type=str, default="tests/example.mp4")
    parser.add_argument("-o", "--output_path", help="output is sorted json array: [[timestamp, slide_num],...]",
                        type=str, default="timestamps.txt")
    parser.add_argument("-t", "--time_period", help="in seconds. Produces timestamps not more often then time_period",
                        type=int, default=1)
    parser.add_argument("-mx", "--max_cntr_area",
                        help="the maximum part of the picture occupied by a slide", type=float, default=0.95)
    parser.add_argument("-mn", "--min_cntr_area",
                        help="the minimum part of the picture occupied by a slide", type=float, default=0.3)
    parser.add_argument("-l", "--lang", help="text detection languages", type=str, default="rus+eng")
    parser.add_argument("-trsh", "--text_len_tresh", help="min ratio (frame detected text length)/(slide text length)",
                        type=float, default=0.1)
    parser.add_argument("--verbose", type=bool, default=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    slide_texts = get_slide_texts(args.slides_path)
    slide_texts = [filter_text(text) for text in slide_texts]
    if args.verbose:
        print("length slides:", len(slide_texts))

    video_imgs = video_to_slide_imgs(args.video_path, args.time_period, args.max_cntr_area, args.min_cntr_area)
    if args.verbose:
        print("length video_imgs:", len(video_imgs))

    # for i, (tmstmp, img) in enumerate(video_imgs.items()):
    #     cv2.imwrite("tmp2/out{}.png".format(i), img)

    result = []

    video_imgs = OrderedDict(sorted(video_imgs.items()))

    for i_img, (tmstmp, img) in enumerate(video_imgs.items()):
        try:
            img_text = get_image_text(img, lang=args.lang)
            img_text = filter_text(img_text)
            best_slide_i, ratio = best_slide(img_text, slide_texts, args.text_len_tresh)  # lower ratio better
            if args.verbose:
                print(str(i_img) + ")", tmstmp, "ms", "slide=", best_slide_i, "ratio=", ratio, '\t', img_text)
            if best_slide_i != -1:  # check if detected img contains text (is slide)
                result.append((tmstmp, best_slide_i))
        except Exception as e:
            print("Error tmstmp=", tmstmp, "ms img_num=", i_img, "img_shape=", img.shape, ":", e)
            cv2.imshow("error img", img)

    with open(args.output_path, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False))
