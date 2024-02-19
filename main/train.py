import datetime
import os
import sys
import time
import tensorflow as tf
from tensorflow.keras import optimizers
from nets import model_train as model
from utils.dataset import data_provider as data_provider
import argparse

# Create the parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--max_steps', type=int, default=60000)
parser.add_argument('--decay_steps', type=int, default=30000)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--moving_average_decay', type=float, default=0.997)
parser.add_argument('--num_readers', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkpoint_path', type=str, default='/content/checkpoints_mlt/')
parser.add_argument('--logs_path', type=str, default='logs_mlt/')
parser.add_argument('--pretrained_model_path', type=str, default='/content/CTPN_NO_SLIM/data/vgg_16.ckpt')
parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--save_checkpoint_steps', type=int, default=2000)

# Parse the arguments
FLAGS = parser.parse_args()

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.keras.Input(shape=[None, None, None, 3], name='input_image')
    input_bbox = tf.keras.Input(shape=[None, 5], name='input_bbox')
    input_im_info = tf.keras.Input(shape=[None, 3], name='input_im_info')

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = optimizers.Adam(learning_rate)

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                 input_im_info)
            grads = opt.get_gradients(total_loss, tf.trainable_variables())
            apply_gradient_op = opt.apply_gradients(zip(grads, tf.trainable_variables()), global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    if FLAGS.pretrained_model_path is not None:
        checkpoint = tf.train.Checkpoint()
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS.pretrained_model_path))
        print("Loaded pretrained model from:", FLAGS.pretrained_model_path)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.compat.v1.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers)
        start = time.time()
        for step in range(restore_step, FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                              feed_dict={input_image: data[0],
                                                         input_bbox: data[1],
                                                         input_im_info: data[2]})

            summary_writer.add_summary(summary_str, global_step=step)

            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.compat.v1.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ml, tl, avg_time_per_step, learning_rate.eval()))
            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))

if __name__ == '__main__':
    main()

