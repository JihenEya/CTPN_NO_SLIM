# coding=utf-8
import os
import shutil
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
import os
import lxml.etree as ET
sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument('--test_data_path', type=str, default='data/demo/')
parser.add_argument('--output_path', type=str, default='data/res/')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint_path', type=str, default='/content/checkpoints_mlt/')
# Parse the arguments
FLAGS = parser.parse_args()



def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


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
    re_im = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


import tensorflow as tf
from tensorflow.keras import optimizers

def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    input_image = tf.keras.Input(shape=[None, None, None, 3], name='input_image')
    input_im_info = tf.keras.Input(shape=[None, 3], name='input_im_info')

    global_step = tf.Variable(0, trainable=False, name='global_step')

    bbox_pred, cls_pred, cls_prob = model.model(input_image)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)

        im_fn_list = get_images()
        data_list = list()
        cordinates_list = list()
        for im_fn in im_fn_list:
            root_element = ET.Element("Image")
            block_count = 0
            line_list = list()
            line_cordinates = list()
            print('===============')
            print(im_fn)
            start = time.time()
            try:
                im = cv2.imread(im_fn)[:, :, ::-1]
            except:
                print("Error reading image {}!".format(im_fn))
                continue

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))

            for i, box in enumerate(boxes):
                tess_image = img[int(box[1]):int(box[5]), int(box[0]):int(box[2])]
                tess_image = cv2.copyMakeBorder(tess_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                text = pytesseract.image_to_string(tess_image, config ='--psm 11', lang='eng', timeout =360)
                line_list.append(text)
                entity_block = ET.Element("block_data", block_number = str(block_count), StartX=str(box[0]), StartY=str(box[1]), EndX=str(box[2]), EndY=str(box[5]))
                clean_string = ''.join(c for c in text if valid_xml_char_ordinal(c))
                entity_block.text = str(clean_string)
                root_element.append(entity_block)
                block_count = block_count + 1
                cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)
            data_list.append(line_list)
            cordinates_list.append(line_cordinates)
            img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])
            
            try:
                    et = ET.ElementTree(root_element)
                    filename_without_ext =  os.path.basename(im_fn).split(".")
                    xmlfilename = filename_without_ext[0] + ".xml"
                    xml_path = os.path.join(os.path.join(FLAGS.output_path), xmlfilename)
                    et.write(xml_path, pretty_print=True, xml_declaration=True, encoding= "utf-8")
            except Exception as e:
                    print("Unable to write to xml")                
                    
                    
def valid_xml_char_ordinal(c):
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
         )                      

if __name__ == '__main__':
    main()

