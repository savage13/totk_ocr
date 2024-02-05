#!/usr/bin/env python3

import os
import sys
import cv2 as cv
import numpy as np
from PIL import Image

if len(sys.argv) < 2:
    print("Usage ocf_file.py moviefile.py")
    print("   Clips out TotK positions from a video file")
    print("   Only every 5th frame is output")
    print("   Make sure the img output directory exists")
    sys.exit(0)

if not os.path.exists("img"):
    print("Error Could not find img output directory")
    sys.exit(0)

frame_skip = 5
wincap = cv.VideoCapture(sys.argv[1])
x = np.array([0.85, 0.89, 0.90, 0.94, 30])
z = np.array([0.88, 0.92, 0.92, 0.95, 0])
y = np.array([0.915, 0.96, 0.88, 0.94, -40])

count = 0
while True:
    wincap.set(cv.CAP_PROP_POS_FRAMES, count)
    ret, img = wincap.read()
    if not ret:
        break
    [h,w,d] = img.shape

    imgs = []
    for ki, v in enumerate([x,z,y]):
        w1 = int(w * v[0])
        w2 = int(w * v[1])
        h1 = int(h * v[2])
        h2 = int(h * v[3])

        tmp = img[ h1:h2, w1:w2 , :]

        if v[4] != 0:
            image_center = tuple(np.array(tmp.shape[1::-1]) / 2)
            rot_mat = cv.getRotationMatrix2D(image_center, v[4], 1.0)
            tmp = cv.warpAffine(tmp, rot_mat, tmp.shape[1::-1],
                                flags=cv.INTER_LINEAR)

        tmp = cv.cvtColor(tmp, cv.COLOR_BGR2HSV)

        r1, r2 = 0, 255
        g1, g2 = 0, 255
        b1, b2 = 120, 180

        lower_blue = np.array([r1, g1, b1])
        upper_blue = np.array([r2, g2, b2])

        mask = cv.inRange(tmp, lower_blue, upper_blue)
        mask = 255 - mask

        im = Image.fromarray(mask)
        png = f"img/{count:08}_{ki}.png"
        im.save(png)

    cv.imshow('img',img)
    print(count)
    count += frame_skip

    sys.stdout.flush()
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break
