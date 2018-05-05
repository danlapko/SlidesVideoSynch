import cv2

from ocr import get_slide_texts, best_slide, get_image_text, filter_text
from video_processing import video_to_slide_imgs

# slides_path = "data/cpp_lec6.pdf"
# video_path = "data/cpp_short.mp4"

slides_path = "data/logic_lec9.pdf"
video_path = "data/logic_short.mp4"

time_period = 1  # sec
max_cntr_area = 0.95
min_cntr_area = 0.3

lang = "rus+eng"
text_len_tresh = 0.1

if __name__ == "__main__":

    slide_texts = get_slide_texts(slides_path)
    slide_texts = [filter_text(text) for text in slide_texts]
    print("len slides:", len(slide_texts))

    video_imgs = video_to_slide_imgs(video_path, time_period, max_cntr_area, min_cntr_area)
    print("len video_imgs:", len(video_imgs))

    result = dict()
    for i_frame, img in video_imgs.items():
        # cool_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),10,30)
        cv2.imwrite("tmp/cool{}_.png".format(i_frame), img)
        img_text = get_image_text(img, lang=lang)
        img_text = filter_text(img_text)
        best_slide_i = best_slide(img_text, slide_texts, text_len_tresh)
        result[i_frame] = (img, best_slide_i)
        print(i_frame, ":", best_slide_i)
        print(img_text)

    # print(result)
