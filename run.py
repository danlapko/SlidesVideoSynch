import json
from collections import OrderedDict
import cv2

from ocr import get_slide_texts, best_slide, get_image_text, filter_text
from video_processing import video_to_slide_imgs

config = {"slides_path": "data/logic_lec9.pdf",
          "video_path": "data/logic_hard_.mp4",
          "output_path": "timestamps_.txt",
          "time_period": 1,  # sec
          "max_cntr_area": 0.95,  # detected slide rectangle max area
          "min_cntr_area": 0.3,
          "lang": "rus+eng",
          "text_len_tresh": 0.1  # min ratio (detected text length)/(slide text length)
          }

if __name__ == "__main__":

    slide_texts = get_slide_texts(config["slides_path"])
    slide_texts = [filter_text(text) for text in slide_texts]
    print("len slides:", len(slide_texts))

    video_imgs = video_to_slide_imgs(config["video_path"], config["time_period"], config["max_cntr_area"],
                                     config["min_cntr_area"])
    print("len video_imgs:", len(video_imgs))

    # for i, (tmstmp, img) in enumerate(video_imgs.items()):
    #     cv2.imwrite("tmp2/out{}.png".format(i), img)

    result = []

    video_imgs = OrderedDict(sorted(video_imgs.items()))

    for i_img, (tmstmp, img) in enumerate(video_imgs.items()):
        try:
            img_text = get_image_text(img, lang=config["lang"])
            img_text = filter_text(img_text)
            best_slide_i, ratio = best_slide(img_text, slide_texts, config["text_len_tresh"])  # lower ratio better
            print(tmstmp, ":", "slide=", best_slide_i, "ratio=", ratio, '\t', img_text)
            if best_slide_i != -1:  # check if detected img contains text (is slide)
                result.append((tmstmp, best_slide_i))
        except Exception as e:
            print("Error tmstmp=", tmstmp, " img_num=", i_img, "img_shape=", img.shape, ":", e)
            cv2.imshow("error img", img)

    with open(config["output_path"], "w") as f:
        f.write(json.dumps(result, ensure_ascii=False))
