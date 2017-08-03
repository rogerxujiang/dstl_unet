# for data feeding of the training of deep neural network

__author__ = 'rj'

'''
Data augmentation includes: 
1. random translation
2. horizontal and vertical flipping
3. random rotation

'''

import pandas as pd
import os
import data_utils
import numpy as np
import cv2



data_dir = '/Users/rogerjiang/code/dstl_unet'

CLASSES = {
    1: 'Bldg',
    2: 'Struct',
    3: 'Road',
    4: 'Track',
    5: 'Trees',
    6: 'Crops',
    7: 'Fast H20',
    8: 'Slow H20',
    9: 'Truck',
    10: 'Car',
}

train_wkt_v4 = pd.read_csv(os.path.join(data_dir, 'data/train_wkt_v4.csv'))
grid_sizes = pd.read_csv(os.path.join(data_dir, 'data/grid_sizes.csv'), skiprows=1, names=['ImageId', 'Xmax', 'Ymin'])
x_crop = 3345
y_crop = 3338
all_train_names = data_utils.all_train_names
no_img = len(data_utils.all_train_names)


def get_all_data():

    image_feature = []
    image_label = []

    for i in range(len(data_utils.all_train_names)):
        image_data = data_utils.ImageData(i)
        image_data.create_train_feature()
        image_data.create_label()

        image_feature.append(image_data.train_feature[:x_crop, :y_crop, :])
        image_label.append(image_data.label[:x_crop, :y_crop, :])

    image_feature = np.stack(image_feature, -1)
    image_label = np.stack(image_label, -1)
    return image_feature, image_label



def input_data(crop_size, channel=0, crop_per_img=1, reflection=True, rotation=8, test=False):

    image_feature, image_label = get_all_data()

    img_ids = range(no_img)

    while True:

        images = []
        labels = []

        if rotation is not None:

            angle = 360. * range(rotation)[np.random.randint(0, rotation)] / rotation
            print 'angle: {}'.format(angle)
            radian = 2. * np.pi * angle / 360.
            print 'radian: {}'.format(radian)

            crop_size_new = int(np.ceil(float(crop_size) * (np.abs(np.sin(radian)) + np.abs(np.cos(radian)))))
            rot_mat = cv2.getRotationMatrix2D((float(crop_size_new) / 2., float(crop_size_new) / 2.), angle, 1.)

            crop_diff = int((crop_size_new - crop_size) / 2.)

            np.random.shuffle(img_ids)

            for i in img_ids:

                for _ in range(crop_per_img):

                    x_base = np.random.randint(0, x_crop-crop_size_new)
                    y_base = np.random.randint(0, y_crop-crop_size_new)

                    print 'x_base for No. {} image'.format(x_base)
                    print 'y_base for No. {} image'.format(y_base)

                    img_crop = image_feature[x_base:x_base+crop_size_new, y_base:y_base+crop_size_new, :, i]
                    label_crop = np.squeeze(image_label[x_base:x_base+crop_size_new, y_base:y_base+crop_size_new, channel, i])

                    img_rot = cv2.warpAffine(img_crop, rot_mat, (crop_size_new, crop_size_new))
                    label_rot = cv2.warpAffine(label_crop, rot_mat, (crop_size_new, crop_size_new))

                    images.append(img_rot[crop_diff:crop_diff+crop_size, crop_diff:crop_diff+crop_size, :])
                    labels.append(label_rot[crop_diff:crop_diff+crop_size, crop_diff:crop_diff+crop_size])


        yield np.stack(images, -1),  np.stack(labels, -1)