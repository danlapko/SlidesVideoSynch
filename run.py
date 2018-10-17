import argparse
import json
from collections import OrderedDict

import cv2

from matcher import best_slide_ml, get_pdf_pages
from ocr import get_slide_texts, best_slide_ocr, get_image_text, filter_text
from video_processing import video_to_slide_imgs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slides_path", help="in pdf format", type=str, default="tests/example.pdf")
    parser.add_argument("-v", "--video_path", type=str, default="tests/example.mp4")
    parser.add_argument("-o", "--output_path", help="output is sorted json array: [[timestamp, slide_num],...]",
                        type=str, default="timestamps.txt")
    parser.add_argument("-t", "--every_frame", help="integer. consider only each every_frame",
                        type=int, default=25 * 5)
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


def remove_repeated_values_from_list(mylist):
    filtered_list = []

    for i, sublist in enumerate(mylist):

        if 0 < i < len(mylist) - 1:
            sublist_prev = mylist[i - 1]
            sublist_next = mylist[i + 1]

            if sublist_prev[1] == sublist[1] and sublist[1] == sublist_next[1]:
                continue

        filtered_list.append(sublist)
    return filtered_list


if __name__ == "__main__":
    args = get_args()
    slide_texts = get_slide_texts(args.slides_path)
    slide_texts = [filter_text(text) for text in slide_texts]
    if args.verbose:
        print("length slides:", len(slide_texts))

    filtered_frames = video_to_slide_imgs(args.video_path, args.every_frame, args.max_cntr_area, args.min_cntr_area)
    if args.verbose:
        print("length filtered_frames:", len(filtered_frames))

    # for i, (tmstmp, img) in enumerate(video_imgs.items()):
    #     cv2.imwrite("tmp2/out{}.png".format(i), img)

    result = []

    print()
    for i_frame, (tmstmp, frame) in enumerate(filtered_frames):
        try:
            frame_text = get_image_text(frame, lang=args.lang)
            frame_text = filter_text(frame_text)
            best_slide_ocr_i, ocr_score = best_slide_ocr(frame_text, slide_texts,
                                                         args.text_len_tresh)  # lower ratio better

            if args.verbose:
                print(f"{i_frame}) {tmstmp/1000/60:.4}min "
                      f"ocr_slide={best_slide_ocr_i } ocr_score={ocr_score:.4} \t{frame_text}")

            if best_slide_ocr_i != -1:  # check if detected img contains text (is slide)
                result.append((tmstmp, best_slide_ocr_i))

        except Exception as e:
            print("Error tmstmp=", tmstmp, "ms img_num=", i_frame, "img_shape=", frame.shape, ":", e)
            cv2.imshow("error img", frame)

    result = remove_repeated_values_from_list(result)
    with open(args.output_path, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False))
