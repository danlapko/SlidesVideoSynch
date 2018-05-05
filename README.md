## Video slides handling
Slides on video detection and timestamps assignment.

Useful for handling lectures videos. You need to have video and pdf slides.  
#### Dependencies
Install opencv and opencv-python

Install tesseract: https://bingrao.github.io/blog/post/2017/07/16/Install-Tesseract-4.0-in-ubuntun-16.04.html

Don't forget to download traindata:

```
wget https://github.com/tesseract-ocr/tessdata/raw/master/eng.traineddata
sudo mv -v eng.traineddata /usr/local/share/tessdata/
wget https://github.com/tesseract-ocr/tessdata/raw/master/rus.traineddata
sudo mv -v rus.traineddata /usr/local/share/tessdata/
```

Install pytesseract:
```
pip install pytesseract
```

Install textract:
```
pip install textract
```

Install editdistance:
```
pip install editdistance
```

#### Launch
 
```
python3 ./run.py --slides_path "tests/example.pdf" --video_path "tests/example.mp4" --output_path "timestamps.txt" --lang "rus+eng"
```

#### Useful ffmpeg commands

`ffmpeg -ss 00:00:03 -i ./example.mp4 -c copy -t 10 ./example.mp4` -cut 10 secs from 00:00:03  
`ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4` - speed up video 2 times 
