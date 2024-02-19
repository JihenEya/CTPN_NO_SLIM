import datetime
import os
import sys
import time

import tensorflow as tf
from tensorflow.keras import optimizers

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
        for im_fn in im_fn_list:
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
                cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)
            img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])

            with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                      "w") as f:
                for i, box in enumerate(boxes):
                    line = ",".join(str(box[k]) for k in range(8))
                    line += "," + str(scores[i]) + "\r\n"
                    f.writelines(line)


if __name__ == '__main__':
    main()
