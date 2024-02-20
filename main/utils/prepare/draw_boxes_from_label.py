import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import cv2

import numpy as np

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')
# if there are no images then no animation can be shown
IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')
IMG_PATH_OUT = os.path.join(os.getcwd(), 'output')
if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            # no image files found
            args.no_animation = True
else:
    args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
        show_animation = True
"""
 Create a ".temp_files/" and "output/" directory
"""
TEMP_FILES_PATH = ".temp_files"
if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
    os.makedirs(TEMP_FILES_PATH)
output_files_path = "output"
if os.path.exists(output_files_path): # if it exist already
    # reset the output directory
    shutil.rmtree(output_files_path)

os.makedirs(output_files_path)

def error(msg):
    print(msg)
    sys.exit(0)

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

"""
 ground-truth
     Load each of the ground-truth files into a temporary ".json" file.
     Create a list of all the class names present in the ground-truth (gt_classes).
"""
def get_GT_data():
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    detection_files_list = glob.glob(DR_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
    # for txt_file in detection_files_list:
        #print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        array_lines = []
        # img = cv2.imread(os.path.join(IMG_PATH, (file_id.split('_')[1] + ".jpg")))
        img = cv2.imread(os.path.join(IMG_PATH, (file_id + ".jpg")))

        # img1, (rh, rw) = resize_image(img)
        # img2 = cv2.resize(img1, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
        for line in lines_list:
            try:
                print(line)
                # xmin, ymin, _, _, xmax, ymax, _, _, original_text = line.split(',', 8)
                xmin, ymin, _, _, xmax, ymax, _, _ = line.split(',', 8)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error(error_msg)
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"bbox":bbox, "used":False})
        # dump bounding_boxes into a ".json" file
        cv2.imwrite(os.path.join(IMG_PATH_OUT, (file_id + ".jpg")), img[:, :, ::-1])
        new_temp_file = TEMP_FILES_PATH + os.sep + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    return gt_files

if __name__ == '__main__':
    get_GT_data()
