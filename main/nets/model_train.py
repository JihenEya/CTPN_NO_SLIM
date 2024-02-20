import tensorflow as tf
from tensorflow.keras import layers
from utils.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_or_size_splits=num_channels, axis=-1)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=-1)


def make_var(name, shape, initializer=None):
      return tf.Variable(initial_value=initializer(shape), trainable=True, name=name)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    shape = tf.shape(net)
    N, H, W, C = shape[0], shape[1], shape[2], shape[3]
    net = tf.reshape(net, [N * H, W, C])
    net.set_shape([None, None, input_channel])

    lstm_fw_cell = layers.LSTMCell(hidden_unit_num)
    lstm_bw_cell = layers.LSTMCell(hidden_unit_num)

    lstm_out, _ = layers.Bidirectional(layers.RNN(lstm_fw_cell), layers.RNN(lstm_bw_cell))(net)
    lstm_out = tf.concat(lstm_out, axis=-1)

    lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

    init_weights = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_avg', distribution='normal')
    init_biases = tf.keras.initializers.Zeros()
    weights = tf.Variable(init_weights(shape=[2 * hidden_unit_num, output_channel]))
    biases = tf.Variable(init_biases(shape=[output_channel]))

    outputs = tf.matmul(lstm_out, weights) + biases

    outputs = tf.reshape(outputs, [N, H, W, output_channel])
    return outputs




def lstm_fc(net, input_channel, output_channel, scope_name):
    shape = tf.shape(net)
    N, H, W, C = shape[0], shape[1], shape[2], shape[3]
    net = tf.reshape(net, [N * H * W, C])

    init_weights = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_avg', distribution='normal')
    init_biases = tf.keras.initializers.Zeros()
    weights = tf.Variable(init_weights(shape=[input_channel, output_channel]))
    biases = tf.Variable(init_biases(shape=[output_channel]))

    output = tf.matmul(net, weights) + biases
    output = tf.reshape(output, [N, H, W, output_channel])
    return output



def model(image):
    # Charger VGG16 sans la partie fully connected en haut
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Ajouter votre propre couche fully connected
    x = base_model(image)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    lstm_output = Bilstm(x, 512, 128, 512, scope_name='BiLSTM')

    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
                          name="cls_prob")

    return bbox_pred, cls_pred, cls_prob

def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    # 'rpn_cls_score', 'gt_boxes', 'im_info'
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        tf.numpy_function(anchor_target_layer_py,
                          [cls_pred, bbox, im_info, [16, ], [16]],
                          [tf.float32, tf.float32, tf.float32, tf.float32])

    rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                      name='rpn_labels')
    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                            name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                   name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                    name='rpn_bbox_outside_weights')

    return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    deltas_abs = tf.abs(deltas)
    smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
    return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
           (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(bbox_pred, cls_pred, bbox, im_info):
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")
    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])
    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
    rpn_label = tf.gather(rpn_label, rpn_keep)
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)
    # box loss
    rpn_bbox_pred = bbox_pred
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]
    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)
    rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), axis=[1])
    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
    model_loss = rpn_cross_entropy + rpn_loss_box
    regularization_losses = tf.losses.get_regularization_losses()
    total_loss = tf.add_n(regularization_losses) + model_loss
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)
    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
