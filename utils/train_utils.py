__author__ = 'rogerjiang'

'''
Purpose:
1. Data augmentation, including:
    1.1 random translation in horizontal and vertical directions
    1.2 horizontal and vertical flipping   
    1.3 random rotation
'''
'''
Class blancing:
    Each class is trained using a different model, weights should be applied
    to the true and false labels if imbalanced.
    
Cross validation can be performed at angles different from the training images.

Loss options:
    1. Jaccard loss
    2. Cross entropy

Optimizer options:
    1. Adam (learning rate drop at around 0.2 of the initial rate for every 
    30 epochs)
    2. NAdam (no improvement over Adam) (50 epochs with a learning rate 
    of 1e-3 and additional 50 epochs with a learning rate of 1e-4. Each epoch 
    was trained on 400 batches, each batch containing 128 image patches (112x112).)

Ensembling:
    1. Arithmetic averaging over different angles
    
Special treatment:
    1. Waterways using NDWI and CCCI).

'''

import pandas as pd
import os
import data_utils
import numpy as np
import cv2
import sys



data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


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
grid_sizes = pd.read_csv(os.path.join(data_dir, 'data/grid_sizes.csv'),
                         skiprows = 1, names = ['ImageId', 'Xmax', 'Ymin'])
x_crop = 3345
y_crop = 3338

test_names = ['6110_1_2', '6110_3_1', '6100_1_3', '6120_2_2']
#train_names = list(set(data_utils.all_train_names) - set(test_names))
train_names = data_utils.all_train_names
test_ids = [data_utils.train_IDs_dict_r[name] for name in test_names]
train_ids = [data_utils.train_IDs_dict_r[name] for name in train_names]


# no_train_img = len(train_names)
# no_test_img = len(test_names)


def generate_train_ids(cl):
    '''
    Create train ids, and exclude the images with no true labels
    :param cl: 
    :return: 
    '''
    df = data_utils.collect_stats()
    df = df.pivot(index = 'ImageId', columns = 'Class', values = 'TotalArea')
    df = df.fillna(0)
    df = df[df[data_utils.CLASSES[cl + 1]] != 0]

    train_names = list(set(list(df.index.get_values())) - set(test_names))

    return [data_utils.train_IDs_dict_r[name] for name in train_names]


def get_all_data(img_ids, train = True):
    '''
    Load all the training feature and label into memory. This requires 35 GB
    memory on Mac and takes a few minutes to finish.
    :return:
    '''
    image_feature = []
    image_label = []
    no_img = len(img_ids)
    phase = ['validation', 'training'][train]
    for i in xrange(no_img):
        id = img_ids[i]
        image_data = data_utils.ImageData(id)
        image_data.create_train_feature()
        image_data.create_label()

        image_feature.append(image_data.train_feature[: x_crop, : y_crop, :])
        image_label.append(image_data.label[: x_crop, : y_crop, :])

        sys.stdout.write('\rLoading {} data: [{}{}] {}%\n'.\
                         format(phase,
                                '=' * i,
                                ' ' * (no_img - i - 1),
                                100 * i / (no_img - 1)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    image_feature = np.stack(image_feature, -1)
    image_label = np.stack(image_label, -1)

    return np.rollaxis(image_feature, 3, 0), np.rollaxis(image_label, 3, 0)


def input_data(crop_size, class_id = 0, crop_per_img = 1,
               reflection = True, rotation = 8, train = True, verbose = False):
    '''
    Returns the training images (feature) and the corresponding labels
    :param crop_size:
    :param class_id:
    :param crop_per_img:
    :param reflection:
    :param rotation:
    :param train:
    :return:
    '''

    # img_ids = generate_train_ids(class_id) if train else test_ids
    img_ids = train_ids if train else test_ids
    no_img = len(img_ids)
    image_feature, image_label = get_all_data(img_ids, train = train)

    while True:

        images = []
        labels = []

        # Rotation angle is assumed to be the same, so that the
        # transformation only needs to be calculated once.
        if not rotation or rotation == 1:
            crop_diff = 0
            crop_size_new = crop_size

        else:
            angle = 360. * np.random.randint(0, rotation) / rotation
            radian = 2. * np.pi * angle / 360.
            if verbose:
                print 'Rotation angle : {0}(degree), {1: 0.2f}(radian)'.\
                    format(int(angle), radian)

            crop_size_new = int(
                np.ceil(float(crop_size) * (abs(np.sin(radian)) +
                                            abs(np.cos(radian)))))
            rot_mat = cv2.getRotationMatrix2D((float(crop_size_new) / 2.,
                                               float(crop_size_new) / 2.),
                                              angle, 1.)
            crop_diff = int((crop_size_new - crop_size) / 2.)

        np.random.shuffle(img_ids)

        for i in xrange(no_img):
            id = img_ids[i]
            for _ in xrange(crop_per_img):

                x_base = np.random.randint(0, x_crop - crop_size_new)
                y_base = np.random.randint(0, y_crop - crop_size_new)
                if verbose:
                    print 'x_base {} for No. {} image'.format(x_base, id)
                    print 'y_base {} for No. {} image'.format(y_base, id)

                img_crop = np.squeeze(image_feature[i, x_base: x_base + crop_size_new,
                           y_base: y_base + crop_size_new, :])
                label_crop = np.squeeze(image_label[i, x_base: x_base + crop_size_new,
                                        y_base: y_base + crop_size_new, class_id])
                if not rotation or rotation == 1:
                    img_rot = img_crop
                    label_rot = label_crop
                else:
                    img_rot = cv2.warpAffine(img_crop, rot_mat,
                                             (crop_size_new, crop_size_new))
                    label_rot = cv2.warpAffine(label_crop, rot_mat,
                                               (crop_size_new, crop_size_new))

                x_step = 1 if not reflection else \
                    [-1, 1][np.random.randint(0, 2)]
                y_step = 1 if not reflection else \
                    [-1, 1][np.random.randint(0, 2)]

                images.append(img_rot[crop_diff: crop_diff + crop_size:,
                              crop_diff: crop_diff + crop_size, :]\
                                  [:: x_step, :: y_step, :])
                labels.append(label_rot[crop_diff: crop_diff + crop_size,
                              crop_diff: crop_diff + crop_size]\
                                  [:: x_step, :: y_step])

        yield np.stack(images, 0), np.stack(labels, 0)


