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
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
from shapely.affinity import scale
import shapely.wkt as wkt

data_dir = '/home/ubuntu/dstl_unet'


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

test_names = ['6110_1_2', '6110_3_1']
train_names = list(set(data_utils.all_train_names) - set(test_names))
test_ids = [data_utils.image_IDs_dict_r[name] for name in test_names]
train_ids = [data_utils.image_IDs_dict_r[name] for name in train_names]


no_train_img = len(train_names)
no_test_img = len(test_names)


def get_all_data(train = True):
    '''
    Load all the training feature and label into memory. This requires 35 GB
    memory on Mac and takes a few minutes to finish.
    :return:
    '''
    image_feature = []
    image_label = []
    img_ids = train_ids if train else test_ids
    no_img = no_train_img if train else no_test_img
    phase = ['validation', 'training'][train]
    for i in xrange(len(img_ids)):
        id = img_ids[i]
        image_data = data_utils.ImageData(id)
        image_data.create_train_feature()
        image_data.create_label()

        image_feature.append(image_data.train_feature[: x_crop, : y_crop, :])
        image_label.append(image_data.label[: x_crop, : y_crop, :])

        sys.stdout.write('\rLoading {} data: [{}{}] {}%'.\
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
    image_feature, image_label = get_all_data(train = train)

    img_ids = train_ids if train else test_ids
    no_img = no_train_img if train else no_test_img


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


def jaccard_index(mask_1, mask_2):
    '''
    Calculate jaccard index between two masks
    :param mask_1:
    :param mask_2:
    :return:
    '''
    assert len(mask_1.shape) == len(mask_2.shape) == 2
    assert 0 <= np.amax(mask_1) <=1
    assert 0 <= np.amax(mask_2) <=1

    intersection = np.sum(mask_1.astype(np.float32) * mask_2.astype(np.float32))
    union = np.sum(mask_1.astype(np.float32) * mask_2.astype(np.float32)) - \
            intersection

    if union == 0:
        return 0.

    return intersection / union


def mask_to_polygons(mask, epsilon = 5, min_area = 1.):
    '''
    Generate polygons from mask
    :param mask:
    :param epsilon:
    :param min_area:
    :return:
    '''
    # find contours, cv2 switches the x-y coordiante of mask to y-x in contours
    # This matches the wkt data in train_wkt_v4, which is desirable for submission
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]

    if not contours:
        return MultiPolygon()

    cnt_children = defaultdict(list)
    child_contours = set()

    assert hierarchy.shape[0] == 1

    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygon filtering by area (remove artifacts)
    all_polygons = []

    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(shell = cnt[:, 0, :],
                           holes = [c[:, 0, :] for c in cnt_children.get(idx, [])
                                    if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])

    return all_polygons


def make_submission():
    '''
    Make submission file from mask files
    :param msk:
    :return:
    '''
    print "Preparing submission file"
    df = pd.read_csv(os.path.join(data_dir, 'data', 'sample_submission.csv'))
    print df.head()
    for id in df.ImageId.unique():

        pred_labels = np.load(os.path.join(data_dir,'msk/10_{}.npy'.format(id)))
        x_max = grid_sizes[grid_sizes.ImageId == id].Xmax.values[0]
        y_min = grid_sizes[grid_sizes.ImageId == id].Ymin.values[0]
        x_scaler, y_scaler = x_max / msk.shape[1], y_min / msk.shape[0]

        for kls in CLASSES:
            msk = np.squeeze(pred_labels[:, :, kls])
            pred_polygons = mask_to_polygons(msk)

            # x-y of mask is switched to y-x in pred_polygons
            scaled_pred_polygons = scale(pred_polygons, xfact = x_scaler,
                                         yfact = y_scaler, origin = (0., 0., 0.))
            dummy = df[df.ImageId == id]
            idx = dummy[dummy.ClassType == kls + 1].index[0]
            df.iloc[idx, 2] = wkt.dumps(scaled_pred_polygons)

            if idx % 100 == 0: print 'Working on image No. {}'.format(idx)

    print df.head()
    df.to_csv(os.path.join(data_dir, 'subm/1.csv'), index = False)

'''

print('Final jaccard',
      final_polygons.intersection(train_polygons).area /
      final_polygons.union(train_polygons).area)

'''