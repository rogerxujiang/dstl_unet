__author__ = 'rj'

'''
This file creates 
'''

import numpy as np
import cv2
from data_utils import rescale
import data_utils



def convert_image_to_display(img):
    '''
    
    :param img: 
    :return: 
    '''

    return 255 * normalize_image(img)



def calculate_warp_matrix(img1, img2):

    img_size = img1.shape
    img_size1 = img2.shape

    nx = img_size[0]
    ny = img_size[1]

    nx_1 = img_size1[0]
    ny_1 = img_size1[0]

    if [nx, ny] != [nx_1, ny_1]:
        img2 = rescale(img1, [nx, ny])

    # Crop the center area to avoid the boundary effect.
    p1 = img1[int(nx * 0.2):int(nx * 0.8), int(ny * 0.2):int(ny * 0.8), :].astype(np.float32)
    p2 = img2[int(nx * 0.2):int(nx * 0.8), int(ny * 0.2):int(ny * 0.8), :].astype(np.float32)

    p1 = normalize_image(p1)
    p2 = normalize_image(p2)

    p1 = np.sum(p1, 2)
    p2 = np.sum(p2, 2)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_mat = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    (cc, warp_mat) = cv2.findTransformECC(p1, p2, warp_mat, warp_mode, criteria)

    print("_align_two_rasters: cc:{}".format(cc))
    print("warp matrix: ")
    print warp_mat

    return warp_mat



def normalize_image(img):
    '''
    Normalize the image (10% to 90%) to [0,1]
    :param img: 
    :return: 
    '''
    img_shape = img.shape

    if len(img_shape) == 3:
        for channel in range(img_shape[2]):
            im_min = np.percentile(img[:,:, channel], 10)
            im_max = np.percentile(img[:,:, channel], 90)
            img[:,:, channel] = (img[:,:, channel].astype(np.float32) - im_min) / (im_max - im_min)
    elif len(img_shape) == 2:
        im_min = np.percentile(img, 10)
        im_max = np.percentile(img, 90)
        img = (img.astype(np.float32) - im_min) / (im_max - im_min)

    return img



if __name__ == '__main__':

    for image_id in range(len(data_utils.all_train_names)):

        img = data_utils.ImageData(image_id)
        images = img.read_image()

        warp_matrix_m = calculate_warp_matrix(images['3'], images['M'])
        warp_matrix_a = calculate_warp_matrix(images['3'], images['A'])

        np.save(open('image_alignment/{}_warp_matrix_m.npz'.format(img.imageId), 'w'), warp_matrix_m)
        np.save(open('image_alignment_new/{}_warp_matrix_a.npz'.format(img.imageId), 'w'), warp_matrix_a)

        #
        img_3 = np.sum(images['3'], 2)
        shape_out = img_3.shape
        img_m_orig = np.sum(data_utils.rescale(images['M'], shape_out), 2)
        img_a_orig = np.sum(data_utils.rescale(images['A'], shape_out), 2)

        img_m_register = np.sum(data_utils.affine_transform(img_m_orig, warp_matrix_m, shape_out), 2)
        img_a_register = np.sum(data_utils.affine_transform(img_a_orig, warp_matrix_a, shape_out), 2)

        img_3_rescale = convert_image_to_display(img_3)
        img_m_orig_rescale = convert_image_to_display(img_m_orig)
        img_a_orig_rescale = convert_image_to_display(img_a_orig)
        img_m_register_rescale = convert_image_to_display(img_m_register)
        img_a_register_rescale = convert_image_to_display(img_a_register)

        empty_channel = np.zeros(shape_out)

        compare_m_orig = np.stack([img_m_orig_rescale, empty_channel, img_3_rescale])
        compare_m_register = np.stack([img_m_register_rescale, empty_channel, img_3_rescale])

        compare_a_orig = np.stack([img_a_orig_rescale, empty_channel, img_3_rescale])
        compare_a_register = np.stack([img_a_register_rescale, empty_channel, img_3_rescale])

        '''
        cv2.imwrite('image_alignment/comparison/{}/3_rescale.png'.format(img.imageId),img_3_rescale)
        cv2.imwrite('image_alignment/comparison/{}/m_orig.png'.format(img.imageId), compare_m_orig)
        cv2.imwrite('image_alignment/comparison/{}/m_register.png'.format(img.imageId), compare_m_register)
        cv2.imwrite('image_alignment/comparison/{}/a_orig.png'.format(img.imageId), compare_a_orig)
        cv2.imwrite('image_alignment/comparison/{}/a_register.png'.format(img.imageId), compare_a_register)
        '''





