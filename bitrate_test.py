# ffmpeg -ss 00:12:50 -i ./moskvin_lec9.mp4 -c copy -t 10 ./moskv.mp4
# ffmpeg -i input.flv -vcodec libx264 -q:v output.mp4
# ffprobe -select_streams v -show_entries packet=size -of compact=p=0:nk=1 ./moskv264.mp4
# ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4
import cv2
import subprocess
from ffprobe3 import FFProbe
import re
import numpy as np
from matplotlib import pyplot as plt


def test1():
    cmnd = "ffprobe -show_frames /home/danila/se_5s/devdays/moskv2_264.mp4 | grep pkt_size"

    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = p.communicate()

    out = out[0].decode('UTF-8')
    out = out.split("pkt_size=")
    out = map(lambda x: x[:-1], out)
    out = filter(lambda x: len(x) > 0, out)
    out = tuple(map(int, out))
    print(out)
    rates = np.array(out)
    N = 25
    # rates = np.convolve(rates, np.ones((N,)) / N, mode='valid')
    plt.plot(rates)
    # plt.show()


def test2():
    cmnd = " ffprobe -select_streams v -show_entries packet=size -of compact=p=0:nk=1 data/lesin.mp4"

    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out = p.communicate()

    out = out[0].decode('UTF-8')
    out = out.split("\n")
    # out = map(lambda x: x, out)
    out = filter(lambda x: len(x) > 0, out)
    out = tuple(map(int, out))
    rates = np.array(out)
    # N = 115
    # rates = np.convolve(rates, np.ones((N,)) / N, mode='valid')
    # deltas = [y - x for x, y in zip(rates, rates[1:])]

    # plt.plot(deltas)
    plt.plot(rates)
    plt.show()


if __name__ == "__main__":
    test2()
