import math
import json
import numpy as np
from numpy.typing import NDArray
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Dropout,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)


class BoxesCalcInfo:
    def __init__(self, max_boxes, iou_threshold, score_threshold):
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold


def mish(x, name):
    return x * tf.math.tanh(tf.math.softplus(x), name=name)


conv_count = 0
def DarknetConv(x, filters, size, strides, batch_norm, activate):
    global conv_count
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005),
               name="conv2d_" + str(conv_count)
               )(x)

    if batch_norm and activate is None:
        x = BatchNormalization(name="batch_norm_" + str(conv_count))(x)

    if batch_norm and activate == "leaky_relu":
        x = BatchNormalization(name="batch_norm_" + str(conv_count))(x)
        x = LeakyReLU(alpha=0.1, name="leaky_relu_" + str(conv_count))(x)

    if batch_norm and activate == "mish":
        x = BatchNormalization(name="batch_norm_" + str(conv_count))(x)
        x = mish(x, name="mish_" + str(conv_count))

    conv_count += 1
    return x


def DarknetResidual(x, filters, activate="leaky_relu"):
    skip = x
    x = DarknetConv(x, filters=filters[0], size=1, strides=1, batch_norm=True, activate=activate)
    x = DarknetConv(x, filters=filters[1], size=3, strides=1, batch_norm=True, activate=activate)
    x = Add()([skip, x])
    return x


def Darknet53Block(x, blocks, filters):
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet53(name=None):
    darknet53_0 = inputs = Input([None, None, 3])
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=32, size=3, strides=1, batch_norm=True,
                              activate="leaky_relu")

    filters = (32, 64)
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=filters[1], size=3, strides=2, batch_norm=True,
                              activate="leaky_relu")
    darknet53_0 = Darknet53Block(darknet53_0, 1, filters)

    filters = (64, 128)
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=filters[1], size=3, strides=2, batch_norm=True,
                              activate="leaky_relu")
    darknet53_0 = Darknet53Block(darknet53_0, 2, filters)

    filters = (128, 256)
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=filters[1], size=3, strides=2, batch_norm=True,
                              activate="leaky_relu")
    darknet53_0 = Darknet53Block(darknet53_0, 8, filters)
    darknet53_2 = darknet53_0

    filters = (256, 512)
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=filters[1], size=3, strides=2, batch_norm=True,
                              activate="leaky_relu")
    darknet53_0 = Darknet53Block(darknet53_0, 8, filters)
    darknet53_1 = darknet53_0

    filters = (512, 1024)
    darknet53_0 = DarknetConv(darknet53_0,
                              filters=filters[1], size=3, strides=2, batch_norm=True,
                              activate="leaky_relu")
    darknet53_0 = Darknet53Block(darknet53_0, 4, filters)

    return tf.keras.Model(inputs, (darknet53_2, darknet53_1, darknet53_0), name=name)


def Darknet19Block(x, filters, pool_size=2):
    x = DarknetConv(x, filters=filters, size=3, strides=1, batch_norm=True, activate="leaky_relu")
    skip = x
    x = MaxPool2D(2, pool_size, 'same')(x)
    return skip, x


def Darknet19(name=None):
    darknet19_0 = inputs = Input([None, None, 3])

    filters = 16
    _, darknet19_0 = Darknet19Block(darknet19_0, filters)

    filters = 32
    _, darknet19_0 = Darknet19Block(darknet19_0, filters)

    filters = 64
    _, darknet19_0 = Darknet19Block(darknet19_0, filters)

    filters = 128
    _, darknet19_0 = Darknet19Block(darknet19_0, filters)

    filters = 256
    darknet19_1, darknet19_0 = Darknet19Block(darknet19_0, filters)

    filters = 512
    _, darknet19_0 = Darknet19Block(darknet19_0, filters, pool_size=1)

    darknet19_0 = DarknetConv(darknet19_0,
                              filters=1024, size=3, strides=1, batch_norm=True,
                              activate="leaky_relu")

    return tf.keras.Model(inputs, (darknet19_1, darknet19_0), name=name)


def DarknetDoublex5(x, filters):
    x = DarknetConv(x,
                    filters=filters, size=1, strides=1, batch_norm=True,
                    activate="leaky_relu")
    x = DarknetConv(x,
                    filters=filters * 2, size=3, strides=1, batch_norm=True,
                    activate="leaky_relu")
    x = DarknetConv(x,
                    filters=filters, size=1, strides=1, batch_norm=True,
                    activate="leaky_relu")
    x = DarknetConv(x,
                    filters=filters * 2, size=3, strides=1, batch_norm=True,
                    activate="leaky_relu")
    x = DarknetConv(x,
                    filters=filters, size=1, strides=1, batch_norm=True,
                    activate="leaky_relu")
    return x


def CSPDarknet53(name=None):
    def CSPResidualBlock(x, blocks, filters):
        x1 = DarknetConv(x, filters=filters[0], size=1, strides=1, batch_norm=True, activate="mish")
        x0 = DarknetConv(x, filters=filters[0], size=1, strides=1, batch_norm=True, activate="mish")
        for _ in range(blocks):
            x0 = DarknetResidual(x0, filters=(filters[2], filters[3]), activate="mish")
        x0 = DarknetConv(x0, filters=filters[0], size=1, strides=1, batch_norm=True,
                         activate="mish")
        x = Concatenate(axis=-1)([x0, x1])
        x = DarknetConv(x, filters=filters[1], size=1, strides=1, batch_norm=True, activate="mish")
        return x

    def CSPSPP(x):
        x = DarknetConv(x,
                        filters=512, size=1, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = DarknetConv(x,
                        filters=1024, size=3, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = DarknetConv(x,
                        filters=512, size=1, strides=1, batch_norm=True,
                        activate="leaky_relu")

        x1 = MaxPool2D(pool_size=13, padding='SAME', strides=1)(x)
        x2 = MaxPool2D(pool_size=9, padding='SAME', strides=1)(x)
        x3 = MaxPool2D(pool_size=5, padding='SAME', strides=1)(x)
        x = Concatenate(axis=-1)([x1, x2, x3, x])

        x = DarknetConv(x,
                        filters=512, size=1, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = DarknetConv(x,
                        filters=1024, size=3, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = DarknetConv(x,
                        filters=512, size=1, strides=1, batch_norm=True,
                        activate="leaky_relu")
        return x

    cspdarknet53_0 = inputs = Input([None, None, 3])

    filters = 32
    cspdarknet53_0 = DarknetConv(cspdarknet53_0,
                                 filters=filters, size=3, strides=1,
                                 batch_norm=True, activate="mish")

    filters = 64
    cspdarknet53_0 = DarknetConv(cspdarknet53_0,
                                 filters=filters, size=3, strides=2,
                                 batch_norm=True, activate="mish")
    cspdarknet53_0 = CSPResidualBlock(cspdarknet53_0, 1,
                                      (filters, filters, filters // 2, filters))

    filters = 128
    cspdarknet53_0 = DarknetConv(cspdarknet53_0, filters=filters, size=3, strides=2,
                                 batch_norm=True, activate="mish")
    cspdarknet53_0 = CSPResidualBlock(cspdarknet53_0, 2,
                                      (filters // 2, filters, filters // 2, filters // 2))

    filters = 256
    cspdarknet53_0 = DarknetConv(cspdarknet53_0, filters=filters, size=3, strides=2,
                                 batch_norm=True, activate="mish")
    cspdarknet53_0 = CSPResidualBlock(cspdarknet53_0, 8,
                                      (filters // 2, filters, filters // 2, filters // 2))
    cspdarknet53_2 = cspdarknet53_0

    filters = 512
    cspdarknet53_0 = DarknetConv(cspdarknet53_0, filters=filters, size=3, strides=2,
                                 batch_norm=True, activate="mish")
    cspdarknet53_0 = CSPResidualBlock(cspdarknet53_0, 8,
                                      (filters // 2, filters, filters // 2, filters // 2))
    cspdarknet53_1 = cspdarknet53_0

    filters = 1024
    cspdarknet53_0 = DarknetConv(cspdarknet53_0, filters=filters, size=3, strides=2,
                                 batch_norm=True, activate="mish")
    cspdarknet53_0 = CSPResidualBlock(cspdarknet53_0, 4,
                                      (filters // 2, filters, filters // 2, filters // 2))

    cspdarknet53_0 = CSPSPP(cspdarknet53_0)

    return tf.keras.Model(inputs, (cspdarknet53_2, cspdarknet53_1, cspdarknet53_0), name=name)


def PANetAndHead(num_classes, anchors, name=None):
    # PANet and Head are one block only because there's no practical way of organizing the
    # weights in the original darknet V4 weights file.
    # as there's no practical reason to seperate PANet and Head - in particular during training
    # we likely freeze only darknet net or subsequent layers - this is it

    def PANet_up(x0, x1, filters):
        x0 = DarknetConv(x0,
                         filters=filters, size=1, strides=1, batch_norm=True,
                         activate="leaky_relu")
        x0 = UpSampling2D(2)(x0)
        x1 = DarknetConv(x1,
                         filters=filters, size=1, strides=1, batch_norm=True,
                         activate="leaky_relu")
        x = Concatenate(axis=-1)([x1, x0])
        x = DarknetDoublex5(x, filters)
        return x

    def PANet_down(x0, x1, filters):
        x0 = DarknetConv(x0,
                         filters=filters, size=3, strides=2, batch_norm=True,
                         activate="leaky_relu")
        x = Concatenate(axis=-1)([x0, x1])
        x = DarknetDoublex5(x, filters)
        return x

    def pa_net_and_head(x_in):
        inputs = (Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:]), Input(x_in[2].shape[1:]))

        pa_net_2 = inputs[0]
        pa_net_1 = inputs[1]
        pa_net_0 = inputs[2]

        # note a conversion happens in PA net. assuming 416 x 416 ...
        # pa_net_0: 13,13 -> 52,52
        # pa_net_1: 26,26 -> 26,26
        # pa_net_2: 52,52 -> 13,13
        # and this pa_net_0 is used for outputs 2, etc

        skip1 = pa_net_0
        pa_net_0 = PANet_up(pa_net_0, pa_net_1, 256)
        skip2 = pa_net_0
        pa_net_0 = PANet_up(pa_net_0, pa_net_2, 128)
        skip3 = pa_net_0
        pa_net_0 = DarknetConv(pa_net_0,
                               filters=256, size=3, strides=1, batch_norm=True,
                               activate="leaky_relu")
        head_2 = DarknetConv(pa_net_0,
                             filters=anchors['yolo_output_2'].shape[0] * (num_classes + 5), size=1,
                             strides=1, batch_norm=False, activate=None)

        pa_net_1 = PANet_down(skip3, skip2, 256)
        skip4 = pa_net_1
        pa_net_1 = DarknetConv(pa_net_1, filters=512, size=3, strides=1, batch_norm=True,
                               activate="leaky_relu")
        head_1 = DarknetConv(pa_net_1,
                             filters=anchors['yolo_output_1'].shape[0] * (num_classes + 5), size=1,
                             strides=1, batch_norm=False, activate=None)

        pa_net_2 = PANet_down(skip4, skip1, 512)
        pa_net_2 = DarknetConv(pa_net_2,
                               filters=1024, size=3, strides=1, batch_norm=True,
                               activate="leaky_relu")
        head_0 = DarknetConv(pa_net_2,
                             filters=anchors['yolo_output_0'].shape[0] * (num_classes + 5), size=1,
                             strides=1, batch_norm=False, activate=None)

        return Model(inputs, (head_2, head_1, head_0), name=name)(x_in)

    return pa_net_and_head


def CSPDarknet29(name=None):
    def CSPDarknet29Block(x, filters):
        skip0 = x
        x = tf.split(x, 2, axis=-1)[1]
        x = DarknetConv(x,
                        filters=filters[0], size=3, strides=1, batch_norm=True,
                        activate="leaky_relu")
        skip1 = x
        x = DarknetConv(x,
                        filters=filters[0], size=3, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = Concatenate(axis=-1)([x, skip1])
        x = DarknetConv(x,
                        filters=filters[1], size=1, strides=1, batch_norm=True,
                        activate="leaky_relu")
        skip2 = x
        x = Concatenate(axis=-1)([skip0, x])
        x = MaxPool2D(2, 2, 'same')(x)

        return skip2, x

    cspdarknet29_0 = inputs = Input([None, None, 3])

    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=32, size=3, strides=2, batch_norm=True,
                                 activate="leaky_relu")
    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=64, size=3, strides=2, batch_norm=True,
                                 activate="leaky_relu")

    filters = (32, 64)
    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=filters[1], size=3, strides=1,
                                 batch_norm=True, activate="leaky_relu")
    _, cspdarknet29_0 = CSPDarknet29Block(cspdarknet29_0, filters=filters)

    filters = (64, 128)
    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=filters[1], size=3, strides=1,
                                 batch_norm=True, activate="leaky_relu")
    _, cspdarknet29_0 = CSPDarknet29Block(cspdarknet29_0, filters=filters)

    filters = (128, 256)
    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=filters[1], size=3, strides=1,
                                 batch_norm=True, activate="leaky_relu")
    cspdarknet29_1, cspdarknet29_0 = CSPDarknet29Block(cspdarknet29_0, filters=filters)

    filters = (256, 512)
    cspdarknet29_0 = DarknetConv(cspdarknet29_0,
                                 filters=filters[1], size=3, strides=1,
                                 batch_norm=True, activate="leaky_relu")

    return tf.keras.Model(inputs, (cspdarknet29_1, cspdarknet29_0), name=name)


def yolo_boxes(pred, anchors, num_classes):
    # # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size_yx = tf.shape(pred)[1:3]
    grid_size_xy = tf.reverse(grid_size_yx, [-1])

    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, num_classes), axis=-1)

    # # works in opencv, but i could never close the entire loop (and we don't do boxes in cv)
    # box_x, box_y, box_w, box_h, objectness, class_probs = tf.split(pred, 5+classes, axis=-1)
    # box_xy = tf.concat((box_x, box_y), axis=-1)
    # box_wh = tf.concat((box_w, box_h), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # tensorflow lite doesn't support tf.size used in tf.meshgrid
    def _meshgrid(n_a, n_b):
        return [
            tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
            tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
        ]

    grid = _meshgrid(grid_size_xy[0], grid_size_xy[1])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size_xy, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2

    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, boxes_calc_info=BoxesCalcInfo(100, 0.5, 0.5)):
    boxes = tf.reshape(tf.constant([], dtype=tf.float32), (tf.shape(outputs[0][0])[0], 0, 4))
    scores = tf.reshape(tf.constant([], dtype=tf.float32), (tf.shape(outputs[0][1])[0], 0, 1))
    class_probs = tf.reshape(tf.constant([], dtype=tf.float32),
                             (tf.shape(outputs[0][2])[0], 0, tf.shape(outputs[0][2])[-1]))

    for output in outputs:
        def reshape_boxes(output):
            return tf.reshape(output[0], (tf.shape(output[0])[0], -1, 4))

        def reshape_scores(output):
            return tf.reshape(output[1], (tf.shape(output[1])[0], -1, 1))

        def reshape_class_probs(output):
            return tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1]))

        boxes = Concatenate(axis=1)([boxes, reshape_boxes(output)])
        scores = Concatenate(axis=1)([scores, reshape_scores(output)])
        class_probs = Concatenate(axis=1)([class_probs, reshape_class_probs(output)])

    boxes = tf.squeeze(boxes, axis=0)
    scores = tf.squeeze(scores, axis=0)
    scores = tf.squeeze(scores, axis=1)
    class_probs = tf.squeeze(class_probs, axis=0)
    classes = tf.argmax(class_probs, axis=1)

    valid_indices, _ = tf.image.non_max_suppression_with_scores(
        boxes=boxes,
        scores=scores,
        max_output_size=boxes_calc_info.max_boxes,
        iou_threshold=boxes_calc_info.iou_threshold,
        score_threshold=boxes_calc_info.score_threshold
    )

    num_boxes = tf.shape(valid_indices)[0]

    scores = tf.gather(scores, valid_indices)
    boxes = tf.gather(boxes, valid_indices)
    classes = tf.gather(classes, valid_indices)

    return num_boxes, boxes, scores, classes


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x,
                        filters=filters * 2, size=3, strides=1, batch_norm=True,
                        activate="leaky_relu")
        x = DarknetConv(x,
                        filters=anchors * (classes + 5), size=1, strides=1, batch_norm=False,
                        activate="leaky_relu")
        x = Lambda(
            lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def YoloV3(num_classes, size_xy, channels, anchors, training=False, do_boxes=True):
    def YoloNeck(filters, name=None):
        def yolo_neck(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x = inputs[0]
                skip = inputs[1]
                x = DarknetConv(x,
                                filters=filters, size=1, strides=1, batch_norm=True,
                                activate="leaky_relu")
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, skip])
            else:
                x = inputs = Input(x_in.shape[1:])

            x = DarknetDoublex5(x, filters)

            return Model(inputs, x, name=name)(x_in)

        return yolo_neck

    inputs = Input([size_xy[1], size_xy[0], channels], name='input')

    backbone_2, backbone_1, backbone_0 = Darknet53(name='yolo_darknet')(inputs)

    has_anchors_0 = ('yolo_output_0' in anchors and
                     anchors['yolo_output_0'] is not None and
                     anchors['yolo_output_0'].size)
    has_anchors_1 = ('yolo_output_1' in anchors and
                     anchors['yolo_output_1'] is not None and
                     anchors['yolo_output_1'].size)
    has_anchors_2 = ('yolo_output_2' in anchors and
                     anchors['yolo_output_2'] is not None and
                     anchors['yolo_output_2'].size)

    yolo_conv_filters = (512, 256, 128)
    outputs = []
    if has_anchors_0 or has_anchors_1 or has_anchors_2:
        filters = yolo_conv_filters[0]
        neck = YoloNeck(filters, name='yolo_conv_0')(backbone_0)
        if has_anchors_0:
            output_0 = YoloOutput(filters,
                                  anchors['yolo_output_0'].shape[0], num_classes,
                                  name='yolo_output_0')(neck)
            outputs.append(output_0)

    if has_anchors_1 or has_anchors_2:
        filters = yolo_conv_filters[1]
        neck = YoloNeck(filters, name='yolo_conv_1')((neck, backbone_1))
        if has_anchors_1:
            output_1 = YoloOutput(filters,
                                  anchors['yolo_output_1'].shape[0], num_classes,
                                  name='yolo_output_1')(neck)
            outputs.append(output_1)

    if has_anchors_2:
        filters = yolo_conv_filters[2]
        neck = YoloNeck(filters, name='yolo_conv_2')((neck, backbone_2))
        if has_anchors_2:
            output_2 = YoloOutput(filters,
                                  anchors['yolo_output_2'].shape[0], num_classes,
                                  name='yolo_output_2')(neck)
            outputs.append(output_2)

    if training or not do_boxes:
        return Model(inputs, outputs, name='yolo_v3')

    boxes = []
    if has_anchors_0:
        boxes_0 = Lambda(lambda output_0_boxes:
                         yolo_boxes(output_0_boxes,
                                    anchors['yolo_output_0'],
                                    num_classes),
                         name='yolo_boxes_0')(output_0)
        boxes.append(boxes_0[:3])

    if has_anchors_1:
        boxes_1 = Lambda(lambda output_1_boxes:
                         yolo_boxes(output_1_boxes,
                                    anchors['yolo_output_1'],
                                    num_classes),
                         name='yolo_boxes_1')(output_1)
        boxes.append(boxes_1[:3])

    if has_anchors_2:
        boxes_2 = Lambda(lambda output_2_boxes:
                         yolo_boxes(output_2_boxes,
                                    anchors['yolo_output_2'],
                                    num_classes),
                         name='yolo_boxes_2')(output_2)
        boxes.append(boxes_2[:3])

    outputs = Lambda(lambda nms_boxes: yolo_nms(nms_boxes), name='yolo_nms')(boxes)

    return Model(inputs, outputs, name='yolo_v3')


def YoloV4(num_classes, size_xy, channels, anchors, training=False, do_boxes=True):
    inputs = Input([size_xy[1], size_xy[0], channels], name='input')
    cspdarknet53_2, cspdarknet53_1, cspdarknet53_0 = CSPDarknet53(name='yolo_darknet')(inputs)
    head_2, head_1, head_0 = PANetAndHead(num_classes, anchors, name='yolo_pa_net')(
        (cspdarknet53_2, cspdarknet53_1, cspdarknet53_0))

    has_anchors_0 = ('yolo_output_0' in anchors and
                     anchors['yolo_output_0'] is not None and
                     anchors['yolo_output_0'].size)
    has_anchors_1 = ('yolo_output_1' in anchors and
                     anchors['yolo_output_1'] is not None and
                     anchors['yolo_output_1'].size)
    has_anchors_2 = ('yolo_output_2' in anchors and
                     anchors['yolo_output_2'] is not None and
                     anchors['yolo_output_2'].size)

    def YoloV4Output(anchors, num_classes, name=None):
        def yolo_v4_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = Lambda(lambda x: tf.reshape(x, (
            -1, x_in.shape[1], x_in.shape[2], anchors, num_classes + 5)))(x)
            return tf.keras.Model(inputs, x, name=name)(x_in)

        return yolo_v4_output

    outputs = []
    if has_anchors_0 or has_anchors_1 or has_anchors_2:
        if has_anchors_0:
            output_0 = YoloV4Output(anchors['yolo_output_0'].shape[0], num_classes,
                                    'yolo_output_0')(head_0)
            outputs.append(output_0)

    if has_anchors_1 or has_anchors_2:
        if has_anchors_1:
            output_1 = YoloV4Output(anchors['yolo_output_1'].shape[0], num_classes,
                                    'yolo_output_1')(head_1)
            outputs.append(output_1)

    if has_anchors_2:
        if has_anchors_2:
            output_2 = YoloV4Output(anchors['yolo_output_2'].shape[0], num_classes,
                                    'yolo_output_2')(head_2)
            outputs.append(output_2)

    if training or not do_boxes:
        return Model(inputs, outputs, name='yolo_v4')

    boxes = []
    if has_anchors_0:
        boxes_0 = Lambda(lambda output_0_boxes:
                         yolo_boxes(output_0_boxes,
                                    anchors['yolo_output_0'],
                                    num_classes),
                         name='yolo_boxes_0')(outputs[0])
        boxes.append(boxes_0[:3])

    if has_anchors_1:
        boxes_1 = Lambda(lambda output_1_boxes:
                         yolo_boxes(output_1_boxes,
                                    anchors['yolo_output_1'],
                                    num_classes),
                         name='yolo_boxes_1')(outputs[1])
        boxes.append(boxes_1[:3])

    if has_anchors_2:
        boxes_2 = Lambda(lambda output_2_boxes:
                         yolo_boxes(output_2_boxes,
                                    anchors['yolo_output_2'],
                                    num_classes),
                         name='yolo_boxes_2')(outputs[2])
        boxes.append(boxes_2[:3])

    outputs = Lambda(lambda nms_boxes: yolo_nms(nms_boxes), name='yolo_nms')(boxes)

    model = Model(inputs, outputs, name='yolo_v4')

    return model


def YoloV3V4Tiny(num_classes, size_xy, channels, anchors, v4=False, training=False, do_boxes=True):
    def YoloNeck(filters, name=None):
        def yolo_neck(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x = inputs[0]
                skip = inputs[1]
                x = DarknetConv(x, filters=filters, size=1, strides=1, batch_norm=True,
                                activate="leaky_relu")
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, skip])
            else:
                x = inputs = Input(x_in.shape[1:])
                x = DarknetConv(x, filters=filters, size=1, strides=1, batch_norm=True,
                                activate="leaky_relu")

            return Model(inputs, x, name=name)(x_in)

        return yolo_neck

    inputs = Input([size_xy[1], size_xy[0], channels], name='input')

    if not v4:
        backbone_1, backbone_0 = Darknet19(name='yolo_darknet')(inputs)
        yolo_conv_filters = (256, 128)
        name_suffix = "v3"
    else:
        backbone_1, backbone_0 = CSPDarknet29(name='yolo_darknet')(inputs)
        yolo_conv_filters = (256, 128)
        name_suffix = "v4"

    has_anchors_0 = ('yolo_output_0' in anchors and
                     anchors['yolo_output_0'] is not None and
                     anchors['yolo_output_0'].size)
    has_anchors_1 = ('yolo_output_1' in anchors and
                     anchors['yolo_output_1'] is not None and
                     anchors['yolo_output_1'].size)

    outputs = []
    if has_anchors_0 or has_anchors_1:
        filters = yolo_conv_filters[0]
        neck = YoloNeck(filters, name='yolo_conv_0')(backbone_0)
        if has_anchors_0:
            output_0 = YoloOutput(filters,
                                  anchors['yolo_output_0'].shape[0], num_classes,
                                  name='yolo_output_0')(neck)
            outputs.append(output_0)

    if has_anchors_1:
        filters = yolo_conv_filters[1]
        neck = YoloNeck(filters, name='yolo_conv_1')((neck, backbone_1))
        if has_anchors_1:
            output_1 = YoloOutput(filters,
                                  anchors['yolo_output_1'].shape[0], num_classes,
                                  name='yolo_output_1')(neck)
            outputs.append(output_1)

    if training or not do_boxes:
        return Model(inputs, outputs, name='yolo_' + name_suffix + '_tiny')

    boxes = []
    if has_anchors_0:
        boxes_0 = Lambda(lambda output_0_boxes:
                         yolo_boxes(output_0_boxes,
                                    anchors['yolo_output_0'],
                                            num_classes),
                         name='yolo_boxes_0')(output_0)
        boxes.append(boxes_0[:3])

    if has_anchors_1:
        boxes_1 = Lambda(lambda output_1_boxes:
                         yolo_boxes(output_1_boxes,
                                    anchors['yolo_output_1'],
                                    num_classes),
                         name='yolo_boxes_1')(output_1)
        boxes.append(boxes_1[:3])

    outputs = Lambda(lambda nms_boxes: yolo_nms(nms_boxes), name='yolo_nms')(boxes)

    return Model(inputs, outputs, name='yolo_' + name_suffix + '_tiny')


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def YoloLoss(anchors, classes=80, ignore_thresh=0.50):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))

        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)

        pred_x = pred_xywh[..., 0]
        pred_y = pred_xywh[..., 1]
        pred_w = pred_xywh[..., 2]
        pred_h = pred_xywh[..., 3]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)

        true_x = (true_box[..., 0] + true_box[..., 2]) / 2
        true_y = (true_box[..., 1] + true_box[..., 3]) / 2
        true_w = true_box[..., 2] - true_box[..., 0]
        true_h = true_box[..., 3] - true_box[..., 1]

        # give higher weights to small boxes
        box_loss_scale = 2 - (true_w * true_h)

        # 3. inverting the pred box equations
        grid_size_x = tf.shape(y_true)[2]
        grid_size_y = tf.shape(y_true)[1]

        grid = tf.meshgrid(tf.range(grid_size_x), tf.range(grid_size_y))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

        true_x = true_x * tf.cast(grid_size_x, tf.float32) - tf.cast(grid[..., 0], tf.float32)
        true_y = true_y * tf.cast(grid_size_y, tf.float32) - tf.cast(grid[..., 1], tf.float32)

        true_w = tf.math.log(true_w / anchors[..., 0])
        true_h = tf.math.log(true_h / anchors[..., 1])
        true_w = tf.where(tf.math.is_inf(true_w), tf.zeros_like(true_w), true_w)
        true_h = tf.where(tf.math.is_inf(true_h), tf.zeros_like(true_h), true_h)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, axis=-1)

        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)

        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        x_loss = obj_mask * box_loss_scale * tf.square(true_x - pred_x)
        y_loss = obj_mask * box_loss_scale * tf.square(true_y - pred_y)
        w_loss = obj_mask * box_loss_scale * tf.square(true_w - pred_w)
        h_loss = obj_mask * box_loss_scale * tf.square(true_h - pred_h)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        # # via binary_crossentropy
        true_class_onehot = tf.one_hot(tf.cast(true_class_idx, tf.int64), depth=classes, axis=-1)
        true_class_binary = tf.reshape(true_class_onehot, (
        tf.shape(y_true)[0], grid_size_y, grid_size_x, tf.shape(y_true)[3], -1, 1))
        pred_class_binary = tf.reshape(pred_class, (
        tf.shape(y_true)[0], grid_size_y, grid_size_x, tf.shape(y_true)[3], -1, 1))
        class_loss = obj_mask * tf.reduce_sum(
            binary_crossentropy(true_class_binary, pred_class_binary), axis=-1)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        x_loss = tf.reduce_sum(x_loss, axis=(1, 2, 3))
        y_loss = tf.reduce_sum(y_loss, axis=(1, 2, 3))
        w_loss = tf.reduce_sum(w_loss, axis=(1, 2, 3))
        h_loss = tf.reduce_sum(h_loss, axis=(1, 2, 3))

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return x_loss + y_loss + w_loss + h_loss + obj_loss + class_loss

    return yolo_loss


def read_anchors(anchors_file_path, is_tiny):
    try:
        anchors_file = open(anchors_file_path)
        anchors_json = json.load(anchors_file)

        if 'anchors' not in anchors_json:
            return None

        anchors = {}
        yolo_outputs = ['yolo_output_0', 'yolo_output_1']
        if not is_tiny:
            yolo_outputs.append('yolo_output_2')

        for yolo_output in yolo_outputs:
            if yolo_output in anchors_json['anchors']:
                anchors[yolo_output] = np.array(anchors_json['anchors'][yolo_output], np.float32)

        return anchors

    except Exception as e:
        print("Error reading ", anchors_file_path, e)
        return None


def darknet_anchors(is_v4, is_tiny, size_xy):
    # these are the original darknet anchors, primarily included for transfer learning

    anchors = {}

    scale_xy = (1., 1.)

    if not is_v4:
        if not is_tiny:
            anchors['yolo_output_0'] = np.array([(116, 90), (156, 198), (373, 326)],
                                                np.float32) * scale_xy
            anchors['yolo_output_1'] = np.array([(30, 61), (62, 45), (59, 119)],
                                                np.float32) * scale_xy
            anchors['yolo_output_2'] = np.array([(10, 13), (16, 30), (33, 23)],
                                                np.float32) * scale_xy
        else:
            anchors['yolo_output_0'] = np.array([(81, 82), (135, 169), (344, 319)],
                                                np.float32) * scale_xy
            anchors['yolo_output_1'] = np.array([(10, 14), (23, 27), (37, 58)],
                                                np.float32) * scale_xy
    else:
        if not is_tiny:
            anchors['yolo_output_0'] = np.array([(142, 110), (192, 243), (459, 401)],
                                                np.float32) * scale_xy
            anchors['yolo_output_1'] = np.array([(36, 75), (76, 55), (72, 146)],
                                                np.float32) * scale_xy
            anchors['yolo_output_2'] = np.array([(12, 16), (19, 36), (40, 28)],
                                                np.float32) * scale_xy
        else:
            anchors['yolo_output_0'] = np.array([(81, 82), (135, 169), (344, 319)],
                                                np.float32) * scale_xy
            anchors['yolo_output_1'] = np.array([(10, 14), (23, 27), (37, 58)],
                                                np.float32) * scale_xy

    return anchors


def get_yolo(num_classes, size_xy, channels, anchors_file, is_v4, is_tiny, training=False,
             do_boxes=True, verbose=1):
    """Load a known test image and ensure we decode as expected. Only for V4 for now. 6 objects should be found.

>>> import cv2 as cv
>>> import utils
>>> num_classes = 80
>>> size_xy = (608, 608)
>>> channels = 3
>>> anchors_file = None
>>> is_v4 = True
>>> is_tiny = False
>>> training = False
>>> do_boxes = True
>>> weights_file ='./unit_test/darknet_weights/darknet_weights_v4'
>>> image_file = './unit_test/unit_test.jpg'
>>> image = cv.imread(image_file)
>>> shape = image.shape
>>> image = cv.resize(image, size_xy)
>>> image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
>>> image = image.astype(np.float32)
>>> image /= 255.0
>>> image = np.expand_dims(image, axis=0)
>>> model, _ = get_yolo(num_classes, size_xy, channels, anchors_file, is_v4, is_tiny, verbose=0)
>>> _ = model.load_weights(weights_file).expect_partial()
>>> predictions = model.predict(image, verbose=0)
>>> predictions = utils.decode_predictions(predictions, shape, .50)
>>> r = [str(prediction) for prediction in predictions]
>>> r
['((array([ 430.4999, 1051.3933], dtype=float32), array([ 798.2594, 1356.4305], dtype=float32)), 41, 0.99679476)', \
'((array([ 41.1644, 793.5317], dtype=float32), array([1447.6521, 1623.8375], dtype=float32)), 63, 0.9934249)', \
'((array([ 397.02045, 1532.2776 ], dtype=float32), array([ 686.5223, 1816.0262], dtype=float32)), 47, 0.98738956)', \
'((array([764.6403  ,  80.470024], dtype=float32), array([2938.619 , 2074.4539], dtype=float32)), 0, 0.971338)', \
'((array([1205.1466 ,  431.77567], dtype=float32), array([1476.7025,  700.6061], dtype=float32)), 74, 0.8820473)', \
'((array([ 576.0404, 1758.7355], dtype=float32), array([ 972.833 , 1892.1376], dtype=float32)), 67, 0.84390086)']
    """

    if anchors_file is None:
        anchors = darknet_anchors(is_v4, is_tiny, size_xy)
    else:
        anchors = read_anchors(anchors_file, is_tiny)

    for anchor in anchors:
        if anchors[anchor] is not None and anchors[anchor].size:
            anchors[anchor] = anchors[anchor] / size_xy
        else:
            anchors[anchor] = anchors[anchor]

    if is_tiny:
        model = YoloV3V4Tiny(num_classes, size_xy, channels, anchors, is_v4,
                             training=training, do_boxes=do_boxes)
    else:
        if not is_v4:
            model = YoloV3(num_classes, size_xy, channels, anchors,
                           training=training, do_boxes=do_boxes)
        else:
            model = YoloV4(num_classes, size_xy, channels, anchors,
                           training=training, do_boxes=do_boxes)

    if verbose:
        for model_layer in model.layers:
            try:
                model_layer.summary()
            except:
                pass
        model.summary()

    return model, anchors


def cv_to_tf(m: NDArray, size_xy: tuple[int, int]):
    m = cv.resize(m, size_xy)
    m = cv.cvtColor(m, cv.COLOR_BGR2RGB)
    m = m.astype(np.float32)
    m /= 255.0
    m = np.expand_dims(m, axis=0)
    return m


def decode_predictions(predictions, image_shape, confidence_threshold):
    num_boxes, boxes, confidences, classes = predictions
    wh = np.flip(image_shape[0:2])
    decoded_predictions = []
    for i in range(num_boxes):
        if not confidences[i] > confidence_threshold:
            continue

        x0y0 = ((np.array(boxes[i][0:2]) * wh).astype(np.float32))
        x1y1 = ((np.array(boxes[i][2:4]) * wh).astype(np.float32))

        decoded_predictions.append(((x0y0, x1y1), classes[i], confidences[i]))

    return decoded_predictions


if __name__ == "__main__":
    import doctest

    doctest.testmod()
