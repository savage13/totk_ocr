#!/usr/bin/env python3

import sys
import os
import argparse
import cv2 as cv
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from windowcapture import WindowCapture

def pair(arg):
    # For simplity, assume arg is a pair of integers
    # separated by a comma. If you want to do more
    # validation, raise argparse.ArgumentError if you
    # encounter a problem.
    return [int(x) for x in arg.split(',')]

parser = argparse.ArgumentParser(description='Process point data')

parser.add_argument('filename', help="data file, read and appended to")
parser.add_argument('-x', type=pair, required=True, help="x limits (E/W)")
parser.add_argument('-y', type=pair, required=True, help="y limits (N/S)")
parser.add_argument('-z', type=pair, required=True, help="z limits (up/down)")
parser.add_argument('-f', type=pair, help="figure size")
parser.add_argument("-o", "--output", help="output plot of current data")
parser.add_argument("-c", "--color", action="store_true", help="color by height (plot only)")
parser.add_argument("-i", "--image", help = "make aspect ratio correct (plot only)")
parser.add_argument("-w", "--window", type=str, help="window title for grabbing")

args = parser.parse_args()

# Set Limits for OCR'ed positions
limits = [args.x, args.y, args.z]

# Clipping of the coordinates
# xmin, xmax, ymin, ymax, rotation angle
x = np.array([0.85, 0.885, 0.90, 0.94, 35])
z = np.array([0.88, 0.92, 0.92, 0.95, 0])
y = np.array([0.915, 0.96, 0.88, 0.94, -40])

# Input filename
data = []
input_file = args.filename
if os.path.exists(input_file):
    data = np.loadtxt(input_file)

# Open file for writing
fout = open(input_file, "a")

# Create Figures
if args.f:
    fig, ax = plt.subplots(figsize=args.f)
else:
    fig, ax = plt.subplots()

# Create Plot
if len(data):
    if args.color:
        plt.scatter(data[:,0], data[:,1],marker='.', c=data[:,2])
    else:
        hl, = plt.plot(data[:,0], data[:,1],'.')
    if not args.output:
        hl2, = plt.plot(data[-1,0], data[-1,1], 'ro')
else:
    hl, = plt.plot([],[],'.')
    hl2, = plt.plot([],[], 'ro')

if args.image:
    plt.axis('image')
plt.ion()
plt.show()

def update(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    hl2.set_xdata(new_data[0])
    hl2.set_ydata(new_data[1])
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.draw()

if args.output:
    plt.savefig(args.output)
    sys.exit(0)

window_name = "Windowed Projector (Scene) - Scene"
if args.window:
    window_name = args.window


wincap = WindowCapture(window_name)

while True:
    # Read image from active window
    img = wincap.read()

    [h,w,d] = img.shape

    pos = [None,None,None]
    err = [None,None,None]
    for ki, v in enumerate([x,z,y]):
        # Bottom Clip Center Number
        w1 = int(w * v[0]) # Left
        w2 = int(w * v[1]) # Right
        h1 = int(h * v[2]) # Top (or is it Bottom)
        h2 = int(h * v[3]) # Bottom

        # Clip text from original image
        tmp = img[ h1:h2, w1:w2 , :]

        # Rotate if necessary
        if v[4] != 0:
            # Rotate about center
            image_center = tuple(np.array(tmp.shape[1::-1]) / 2)
            rot_mat = cv.getRotationMatrix2D(image_center, v[4], 1.0)
            tmp = cv.warpAffine(tmp, rot_mat, tmp.shape[1::-1],
                                flags=cv.INTER_LINEAR)

        # Convert from RGB to HSV
        tmp = cv.cvtColor(tmp, cv.COLOR_BGR2HSV)

        r1, r2 = 0, 255
        g1, g2 = 0, 255
        b1, b2 = 120, 180

        # Threshold in HSV space
        lower_blue = np.array([r1, g1, b1]) 
        upper_blue = np.array([r2, g2, b2]) 

        mask = cv.inRange(tmp, lower_blue, upper_blue) 
        # Flip black/white bits
        mask = 255 - mask

        config = "--psm 6 --oem 3 outputbase digits"
        out = pytesseract.image_to_string(mask, lang="eng",
                                          config = config)
        out = out.strip()
        cv.imshow(str(ki), mask)
        try :
            val = int(out)
            pos[ki] = val
        except:
            err[ki] = out
            pass

    # Check to see if all positions were read
    flag = not None in pos

    # Check to see if positions are within the limits
    if flag:
        for ki, v in enumerate(pos):
            if not (limits[ki][0] <= v <= limits[ki][1]):
                print("OOB", limits[ki], v, file=sys.stderr)
                flag = False

    # If OCR is ok, output to file and update plot
    if flag:
        pt = " ".join(map(str, pos))
        update(hl, pos)
        print(pt, file=fout)
        fout.flush()
    else:
        print("ERR", pos, err, file=sys.stderr)


    # Display individual coordinates 
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

