import pytesseract
from pytesseract import Output
import textract
import cv2
from difflib import SequenceMatcher
import editdistance


def get_slide_texts(path):
    text = textract.process(path).decode('utf-8')
    slides = text.split("\u000c")
    return slides


def filter_text(text):
    res = filter(lambda x: "а" <= x <= "я" or "А" <= x <= "Я" or
                           "a" <= x <= "z" or "A" <= x <= "Z", text)
                            # or x == " " or x == "\n", text)
    res = "".join(res).lower()
    return res


def get_image_text(img, lang="rus+eng"):
    d = pytesseract.image_to_boxes(img, lang=lang, output_type=Output.DICT)
    return "".join(d['char'])


def strings_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def best_slide_ocr(img_text, slide_texts, len_text_tresh):
    best_score = float('inf')
    best_score_cnt = 1
    best_i = -1
    for i, slide in enumerate(slide_texts):
        if len(slide) == 0 or len(img_text) / len(slide) <= len_text_tresh:
            continue
        score = (editdistance.eval(img_text, slide) - (len(slide) - len(img_text))) / len(img_text)
        # dist = strings_similarity(img_text, slide)
        if score == best_score:
            best_score_cnt += 1

        if score < best_score:
            best_i = i
            best_score = score
            best_score_cnt = 1

    if best_score_cnt > 1:
        return -1, best_score
    return best_i, best_score


if __name__ == "__main__":
    pdf_path = "data/cpp_lec6.pdf"
    path = "data/recognized.png"

    slides = get_slide_texts(pdf_path)

    for i, slide in enumerate(slides):
        slides[i] = filter_text(slide)

    img = cv2.imread(path)
    img_text = get_image_text(img)
    img_text = filter_text(img_text)

    best_slide_i, ratio = best_slide_ocr(img_text, slides, 0.1)
    print("======== best_slide=", best_slide_i, "ratio=", ratio, "=======")
    print(slides[best_slide_i])
    print(img_text)
