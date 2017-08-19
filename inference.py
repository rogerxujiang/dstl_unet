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



if __name__ == '__main__':
    hypes = './hypes/hypes.json'
    with open(hypes, 'r') as f:
        H = simplejson.load(f)
        H['batch_size'] = 1
        H['pad'] = 100
        H['x_width'] = 1920
        H['x_height'] = 1920
        H['print_iter'] = 100
        print_iter = H['print_iter']
        num_channel = H['num_channel']
        x_width = H['x_width']
        x_height = H['x_height']
        batch_size = H['batch_size']
        class_type = H['class_type']
        pad = H['pad']
        [crop_width, crop_height] = [1700, 1700]
        class_type = H['class_type']
        log_dir = H['log_dir']

    img_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_width, x_height, 16])
    logits, pred = train.build_pred(img_in, H, 'test')


    def test_input(img, img_size):

        [img_width, img_height] = img_size
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


    def stitch_mask(img_stack, img_size, feat_shape):

        mask = np.zeros([8, img_size[0], img_size[1]])
        [img_width, img_height] = img_size

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
                        [pad: pad + feat_shape[idx][0], pad: pad + feat_shape[idx][1]]

                    img_stack[idx] = img_stack[idx][::x_step, ::y_step]

                    if feat_trans == 1:
                        img_stack[idx] = np.rollaxis(img_stack[idx], 1, 0)

                    mask[quarter, x_start: x_end, y_start: y_end] = img_stack[idx]

                    quarter += 1
                    idx += 1

        return np.squeeze((np.mean(mask, axis=0) > 0.5).astype(np.int))


    sys.stdout.write('\n')
    sys.stdout.write('#' * 80 + '\n')
    sys.stdout.write("Preparing submission file for class type {}\n".format(class_type).ljust(55, '#').rjust(80, '#') + '\n')
    sys.stdout.write('#' * 80 + '\n')
    sys.stdout.write('\n')
    sys.stdout.flush()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    df = pd.read_csv('data/sample_submission.csv')

    with tf.Session(config=config) as sess:

        saver.restore(sess, save_path='log_dir//ckpt')
        start_time = time.time()
        sys.stdout.write('\n')

        for idx, row in df.iterrows():

            if row[1] == class_type + 1:
                img_id = data_utils.test_IDs_dict_r[row[0]]
                img_data = data_utils.ImageData(img_id, phase='test')
                img_data.load_image()
                img_data.create_train_feature()

                mask_stack = []
                shape_stack = []
                for feat_shape, img in test_input(img_data.train_feature, img_data.image_size):
                    predictions, = sess.run([pred], feed_dict={
                        img_in: np.reshape(img, [batch_size, x_width, x_height, num_channel])})
                    mask_stack.append(predictions)
                    shape_stack.append(feat_shape)

                mask = stitch_mask(mask_stack, img_data.image_size, shape_stack)

                polygons = data_utils.mask_to_polygons(mask=mask, img_id=img_id, test=True, epsilon=1)

                df.iloc[idx, 2] = wkt.dumps(polygons) if len(polygons) else 'MULTIPOLYGON EMPTY'

            if idx % print_iter == 0:
                str1 = 'Working on Image No. {} Class {}: '.format(idx, class_type)
                str2 = 'Time / image: {0:.2f} (mins); '. \
                    format((time.time() - start_time) / 60. / print_iter if idx else 0)
                sys.stdout.write(str1 + str2)
                sys.stdout.flush()
                start_time = time.time()

    sys.stdout.write('\n')
    print df.head()
    if not os.path.exists('./submission'):
        os.makedirs('./submission')
    df.to_csv(os.path.join('.', 'submission/class_{}.csv'.format(class_type)), index=False)