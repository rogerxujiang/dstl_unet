__author__ = 'rogerjiang'

'''
This file performs the inference based on the learned mode
parameters (checkpoint). Each image is cut into 4 quarters due
to the limit of gpu memory. Vertical and horizonta reflections,
and [0, 90, 180, 270] degrees rotations are performed for each
quarter, and the outputs of sigmoid predictions are later
arithmetically averaged.
'''

import tensorflow as tf
import simplejson
from utils import data_utils, train_utils
import os
import numpy as np
import cv2
import train
import pandas as pd
from shapely import wkt
import time
import sys



def test_input(img, img_size, H):
    '''
    This function cuts each image into 4 quarters:
    ([upper, lower] * [left, right]), performs vertical and horizontal
    reflections, and [0, 90, 180, 270] degrees rotations for each quarter.
    It yields (4 * 2 * 4 =)32 images.
    :param img:
    :param img_size:
    :return:
    '''
    [img_width, img_height] = img_size
    [crop_width, crop_height] = H['crop_size']
    pad = H['pad']
    x_width = H['x_width']
    x_height = H['x_height']

    for [x_start, x_end, y_start, y_end] in [
        [0, crop_width, 0, crop_height],
        [0, crop_width, crop_height, img_height],
        [crop_width, img_width, 0, crop_height],
        [crop_width, img_width, crop_height, img_height],
    ]:
        feature = img[x_start: x_end, y_start:y_end, :]
        for feat_trans in [feature, np.rollaxis(feature, 1, 0)]:
            for [x_step, y_step] in [[1, 1], [-1, 1], [1, -1], [-1, -1]]:
                feature_w_padding = cv2.copyMakeBorder(
                    feat_trans[::x_step, ::y_step, :],
                    pad, x_width - pad - feat_trans.shape[0],
                    pad, x_height - pad - feat_trans.shape[1],
                    cv2.BORDER_REFLECT_101)
                yield feat_trans.shape, feature_w_padding



def pred_for_each_quarter(sess, img_in, pred, img_data, H):
    '''

    :param sess:
    :param img_in:
    :param pred:
    :param img_data:
    :param H:
    :return:
    '''
    mask_stack, shape_stack = [], []
    for feat_shape, img in test_input(
            img_data.train_feature, img_data.image_size, H):
        predictions, = sess.run(
            [pred],
            feed_dict={img_in: np.reshape(img,
                                          [batch_size,
                                           x_width,
                                           x_height,
                                           num_channel])})
        mask_stack.append(predictions)
        shape_stack.append(feat_shape)
    return mask_stack, shape_stack

def stitch_mask(img_stack, img_size, feat_shape, H):
    '''
    img_stack is the stack of pixel-wise inference of input images from
    test_input. This function reverts the reflection and rotations and
    stitches the 4 quarters together.
    :param img_stack:
    :param img_size:
    :param feat_shape:
    :return:
    '''
    mask = np.zeros([8, img_size[0], img_size[1]])
    [img_width, img_height] = img_size
    [crop_width, crop_height] = H['crop_size']
    pad = H['pad']

    idx = 0
    for [x_start, x_end, y_start, y_end] in [
        [0, crop_width, 0, crop_height],
        [0, crop_width, crop_height, img_height],
        [crop_width, img_width, 0, crop_height],
        [crop_width, img_width, crop_height, img_height],
    ]:
        quarter = 0
        for feat_trans in range(2):
            for [x_step, y_step] in [[1, 1], [-1, 1], [1, -1], [-1, -1]]:

                img_stack[idx] = img_stack[idx] \
                    [pad: pad + feat_shape[idx][0],
                                 pad: pad + feat_shape[idx][1]]

                img_stack[idx] = img_stack[idx][::x_step, ::y_step]

                if feat_trans == 1:
                    img_stack[idx] = np.rollaxis(img_stack[idx], 1, 0)

                mask[quarter, x_start: x_end, y_start: y_end] = img_stack[idx]

                quarter += 1
                idx += 1

    return np.squeeze((np.mean(mask, axis=0) > 0.5).astype(np.int))



if __name__ == '__main__':
    hypes = './hypes/hypes.json'
    with open(hypes, 'r') as f:

        H = simplejson.load(f)

        H['batch_size'] = 1
        H['pad'] = 100
        H['x_width'] = 1920
        H['x_height'] = 1920
        H['print_iter'] = 100
        H['save_iter'] = 500
        H['crop_size'] = [1700, 1700]

        print_iter = H['print_iter']
        num_channel = H['num_channel']
        x_width = H['x_width']
        x_height = H['x_height']
        batch_size = H['batch_size']
        class_type = H['class_type']
        pad = H['pad']
        class_type = H['class_type']
        log_dir = H['log_dir']
        save_iter = H['save_iter']
        # Crop area for each inference, and this is limited by memory of k80 gpu
        [crop_width, crop_height] = H['crop_size']

    img_in = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size, x_width, x_height, 16])
    logits, pred = train.build_pred(img_in, H, 'test')

    sys.stdout.write('\n')
    sys.stdout.write('#' * 80 + '\n')
    sys.stdout.write("Preparing submission file for class type {}".\
                     format(class_type).ljust(55, '#').rjust(80, '#') + '\n')
    sys.stdout.write('#' * 80 + '\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    df = pd.read_csv('data/sample_submission.csv')

    if not os.path.exists('./submission'):
        os.makedirs('./submission')

    with tf.Session(config=config) as sess:

        saver.restore(sess, save_path= 'log_dir/path_to_ckpt/ckpt/ckpt-9000')
        start_time = time.time()
        sys.stdout.write('\n')

        for idx, row in df.iterrows():

            if row[1] == class_type + 1:
                img_id = data_utils.test_IDs_dict_r[row[0]]
                img_data = data_utils.ImageData(img_id, phase='test')
                img_data.load_image()
                img_data.create_train_feature()

                mask_stack, shape_stack = pred_for_each_quarter(
                    sess, img_in, pred, img_data, H)
                mask = stitch_mask(mask_stack, img_data.image_size, shape_stack, H)

                polygons = data_utils.mask_to_polygons(
                    mask=mask, img_id=img_id, test=True, epsilon=1)

                df.iloc[idx, 2] = \
                    wkt.dumps(polygons) if len(polygons) else 'MULTIPOLYGON EMPTY'

            if idx % print_iter == 0:
                str1 = 'Working on Image No. {} Class {}: '.format(idx, class_type)
                str2 = 'Time / image: {0:.2f} (mins); '. \
                    format((time.time() - start_time) / 60. / print_iter \
                                                      if idx else 0)
                sys.stdout.write(str1 + str2 + '\n')
                sys.stdout.flush()
                start_time = time.time()

            # Save some intermediate results in case of interruption.
            if idx % save_iter == 0:
                df.to_csv(
                    os.path.join('.','submission/class_{}.csv'.format(class_type)),
                    index=False)

    sys.stdout.write('\n')
    print df.head()

    df.to_csv(
        os.path.join('.', 'submission/class_{}.csv'.format(class_type)),
        index=False)