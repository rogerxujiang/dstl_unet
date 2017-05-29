__author__ = 'rj'


'''
Purposes:
 1. training data visualization
 2. evaluation of training data augamentation
 3. evaluation of the trained model
'''


'''
Notes on the data files:

train_wkt_v4.csv: training labels with ImageId, ClassType, MultipolygonWKT

train_geoson_v3 (similar to train_wkt_v4.csv): training labels with ImageId (folder name), ClassType (detailed, name of .geojson files), Multipolygon (data of .geojson files, also contains detailed ClassType information)

grid_size.csv: sizes of all images with ImageId, 0<Xmax<1, -1<Ymin<0 (size of images, assuming origin (0,0) is at the upper left corner)

three_band: all 3-band images, in name of ImageId.tif

sixteen_band: all 16-band images, in name of ImageId_{A,M,P}.tif

sample_submission.csv: submission with ImageId, ClassType, MultipolygonWKT

-------------

Basically, the combination of ClassType and MultipolygonWKT gives the voxel-wise class labels.

The 'three_band' and 'sixteen_band' folders are the input for training.

ImageId connects the class labels with the training data.

MultipolygonWKT is relative position in the figure and can be converted to pixel coordinate with the grid_size (Xmax, Ymin)

There is slightly mismatch between the three_band and sixteen_band data, such that they should be aligned.

'''


import tifffile
import shapely.wkt as wkt
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from matplotlib.patches import Patch
import random
from matplotlib import cm
from shapely import affinity
import sys

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
COLORS = {
    1: '0.7',
    2: '0.4',
    3: '#b35806',
    4: '#dfc27d',
    5: '#1b7837',
    6: '#a6dba0',
    7: '#74add1',
    8: '#4575b4',
    9: '#f46d43',
    10: '#d73027',
}
ZORDER = {
  1: 6,
  2: 5,
  3: 4,
  4: 1,
  5: 3,
  6: 2,
  7: 7,
  8: 8,
  9: 9,
  10: 10,
}



_df = pd.read_csv(data_dir+'/data/train_wkt_v4.csv', names=['ImageId', 'ClassId', 'MultipolygonWKT'], skiprows=1)
_df1 = pd.read_csv(data_dir+'/data/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
#duplicates = ['6110_1_2', '6110_3_1']
duplicates = []

train_wkt_v4 = _df[np.invert(np.in1d(_df.ImageId, duplicates))]
grid_sizes = _df1[np.invert(np.in1d(_df1.ImageId, duplicates))]

all_train_names = train_wkt_v4.ImageId.unique()

image_IDs_dict = dict(zip(np.arange(all_train_names.size),all_train_names))



def resize(im, shape_out):
    '''
    
    :param im: 
    :param shape_out: 
    :return: resized image
    '''
    return cv2.resize(im, (shape_out[1], shape_out[0]), interpolation=cv2.INTER_CUBIC)



def affine_transform(img, warp_matrix, out_shape):
  """
  Apply affine transformations with warp_matrix, perform interpolation as
  needed
  :param img:
  :param warp_matrix:
  :param out_shape: desired output dimension
  :return:
  """
  new_img = cv2.warpAffine(img, warp_matrix, (out_shape[1], out_shape[0]),
                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
  new_img[new_img == 0] = np.average(new_img)
  return new_img



def get_polygon_list(imageId, classType):
    '''
    
    :param imageId: 
    :param classType: 
    :return: a list of polygon list for a specific imageId and classType
    '''
    all_polygon = train_wkt_v4[train_wkt_v4.ImageId == imageId]
    polygon = all_polygon[all_polygon.ClassId == classType].MultipolygonWKT

    polygon_list = None

    if len(polygon) > 0:
        assert len(polygon) == 1
        polygon_list = wkt.loads(polygon.values[0])

    return polygon_list



def convert_coordinate_to_raster(coords, img_size, xymax):
    '''
    Converts the coordinates of contours from relative positions to 
    image (raster) coordinates.
    :param coords: 
    :param img_size: 
    :param xymax: 
    :return: 
    '''
    xmax, ymax = xymax
    width, height = img_size

    coords[:, 0] *= height+1
    coords[:, 1] *= width+1

    coords[:, 0] /= xmax
    coords[:, 1] /= ymax

    coords = np.round(coords).astype(np.int32)

    return coords



def genetate_contours(polygonlist, img_size, xymax):
    '''
    Generate coordinates of contours from polygonlist
    :param polygonlist: 
    :param img_size: 
    :param xymax: 
    :return: 
    '''
    perim_list = []
    inter_list = []

    if polygonlist is None:
        return None

    for i in range(len(polygonlist)):
        poly = polygonlist[i]
        perim = np.array(list(poly.exterior.coords))
        perim_raster = convert_coordinate_to_raster(perim, img_size, xymax)
        perim_list.append(perim_raster)

        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_raster = convert_coordinate_to_raster(interior, img_size, xymax)
            inter_list.append(interior_raster)
    return perim_list, inter_list



def generate_mask_from_contours(img_size, perim_list, inter_list, class_id=0):
    '''
    Create mask for a specific class
    :param img_size: 
    :param perim_list: 
    :param inter_list: 
    :param class_id: 
    :return: 
    '''

    mask = np.zeros(img_size, np.uint8)

    if perim_list is None:
        return mask

    cv2.fillPoly(mask, perim_list, class_id)
    cv2.fillPoly(mask, inter_list, 0)
    return mask



def plot_polygon(polygon_list, ax, scaler=None, alpha=0.7):

    legend_list = []
    for cl in CLASSES:
        legend_list.append(Patch(color=COLORS[cl], label='{}: ({})'.format(CLASSES[cl], len(polygon_list[cl]))))
        for polygon in polygon_list[cl]:
            if scaler is not None:
                polygon_rescale = affinity.scale(polygon, xfact=scaler[0], yfact=scaler[1], origin=[0.,0.,0.])
            else:
                polygon_rescale = polygon
            patch = PolygonPatch(polygon=polygon_rescale, color=COLORS[cl], lw=0, alpha=alpha, zorder=ZORDER[cl])
            ax.add_patch(patch)
    ax.autoscale_view()
    ax.set_title('Objects')
    ax.set_xticks([])
    ax.set_yticks([])
    return legend_list



def plot_image(img, ax, image_id, image_key, selected_channel=None):
    '''
    Plot an image into axis ax with specific axis labels and titles
    :param img: 
    :param ax: 
    :param image_id: 
    :param image_key: 
    :param selected_channel: 
    :return: 
    '''
    title_suffix = ''
    if selected_channel is not None:
        img = img[:,:, selected_channel]
        title_suffix = '('+','.join(repr(i) for i in selected_channel)+')'
    ax.imshow(img)
    ax.set_title(image_id+'-'+image_key+title_suffix)
    ax.set_xlabel(img.shape[0])
    ax.set_ylabel(img.shape[1])
    ax.set_xticks([])
    ax.set_yticks([])



def plot_overlay(img, ax, image_id, image_key, polygon_list, scaler=[1.,1.]):
    ax.imshow(scale_percentile(rgb2gray(img)), cmap=cm.gray, vmax=1., vmin=0.)
    ax.set_xlabel(img.shape[0])
    ax.set_ylabel(img.shape[1])
    plot_polygon(polygon_list, ax, scaler=scaler, alpha=0.8)
    ax.set_title(image_id+'-'+image_key+'-Overlay')



def scale_percentile(img):
    '''
    Scale an image's 1%-99% into 0.-1.
    :param img: 
    :return: 
    '''
    orig_shape = img.shape
    if len(orig_shape) == 3:
        img = np.reshape(img, [orig_shape[0]*orig_shape[1], orig_shape[2]]).astype(np.float32)
    elif len(orig_shape) == 2:
        img = np.reshape(img, [orig_shape[0] * orig_shape[1]]).astype(np.float32)
    mins = np.percentile(img, 1, axis=0)
    maxs = np.percentile(img, 99, axis=0) - mins

    img = (img - mins)/maxs

    img.clip(0.,1.)
    img = np.reshape(img, orig_shape)

    return img



def rotate():
    return 0



def crop(img, crop_area):
    '''
    Crop out and return an area from an image.
    :param img: 
    :param crop_area: 
    :return: 
    '''
    width, height = img.shape[0], img.shape[1]
    x_lim = crop_area[0].astype(np.int)
    y_lim = crop_area[1].astype(np.int)

    assert x_lim[0] >= 0 and x_lim[1] > x_lim[0]
    assert x_lim[1] <= width
    assert y_lim[0] >= 0 and y_lim[1] > y_lim[0]
    assert y_lim[1] <= height

    return img[x_lim[0]:x_lim[1], y_lim[0]:y_lim[1]]


def get_image_area(image_id):
    '''
    returns the area of an image
    :param image_id: 
    :return: 
    '''
    xmax = grid_sizes[grid_sizes.ImageId == image_id].Xmax.values[0]
    ymin = grid_sizes[grid_sizes.ImageId == image_id].Ymin.values[0]

    return np.abs(xmax * ymin)
# 1. histogram for each class across training examples
# 2. percentage of pixels for a certain class, averaged over all images
# 3.
def image_stat(image_id):
    '''
    returns the statistics of an image as a pandas dataframe
    :param image_id: 
    :return: 
    '''
    counts, totalArea, meanArea, stdArea = {}, {}, {}, {}
    image_area = get_image_area(image_id)

    for cl in CLASSES:
        polygon_list = get_polygon_list(image_id, cl)
        counts[cl] = len(polygon_list)
        if len(polygon_list) > 0:
            totalArea[cl] = np.sum([poly.area for poly in polygon_list]) / image_area * 100.
            meanArea[cl] = np.mean([poly.area for poly in polygon_list]) / image_area * 100.
            stdArea[cl] = np.std([poly.area for poly in polygon_list]) / image_area * 100.


    return pd.DataFrame({'CLASS': CLASSES, 'counts': counts, 'totalArea': totalArea, 'meanArea': meanArea, 'stdArea': stdArea})



def collect_stats():
    '''
    create the statistics of all images
    :return: 
    '''
    stats = []
    for image_no, image_id in enumerate(all_train_names):
        print 'Working on image No. {}: {} \n'.format(image_no, image_id)
        stat = image_stat(image_id)
        stat['ImageId'] = image_id
        stats.append(stat)
        print 'prepare image patches [                    ]',
        print '\b' * 22,
        sys.stdout.flush()
    return pd.concat(stats)



def plot_stats():
    



def rgb2gray(rgb):
    '''
    convert rgb image to grey scale
    :param rgb: 
    :return: 
    '''
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])



class ImageData():

    def __init__(self, image_id):

        self.imageId = image_IDs_dict[image_id]
        self.stat = image_stat(self.imageId)
        self.three_band_image = None
        self.sixteen_band_image = None
        self.image = None
        self.image_size = None
        self._xymax = None
        self.label = None
        self.crop_image = None


    def load_image(self):
        im = self.image_stack()
        self.three_band_image = im[:,:,0:3]
        self.sixteen_band_image = im[:,:,3:]
        self.image = im
        self.image_size = np.shape(im)[0:2]
        xmax = grid_sizes[grid_sizes.ImageId == self.imageId].Xmax.values[0]
        ymax = grid_sizes[grid_sizes.ImageId == self.imageId].Ymin.values[0]
        self._xymax = [xmax, ymax]


    def get_image_path(self):
        '''
        Return the path of images
        :param imageId: 
        :return: 
        '''
        return {
            '3': '{}/data/three_band/{}.tif'.format(data_dir, self.imageId),
            'A': '{}/data/sixteen_band/{}_A.tif'.format(data_dir, self.imageId),
            'M': '{}/data/sixteen_band/{}_M.tif'.format(data_dir, self.imageId),
            'P': '{}/data/sixteen_band/{}_P.tif'.format(data_dir, self.imageId),
        }


    def read_image(self):
        '''
        
        :return: All images as a dictionary.
        '''
        images = {}
        path = self.get_image_path()
        for key in path:
            im = tifffile.imread(path[key])
            if key != 'P':
                print
                images[key] = np.transpose(im, (1, 2, 0))
            elif key == 'P':
                images[key] = im

        im3 = images['3']
        ima = images['A']
        imm = images['M']

        [nx, ny, _] = im3.shape

        images['A'] = resize(ima, [nx, ny])
        images['M'] = resize(imm, [nx, ny])

        return images


    def image_stack(self):
        '''
        Resample all images to the highest resolution
        Align all images.
        :return: image: Nx x Ny x Nchannel
        '''

        images = self.read_image()

        im3 = images['3']
        ima = images['A']
        imm = images['M']

        [nx, ny, _] = im3.shape

        warp_matrix_a = np.load((data_dir+'/utils/image_alignment/{}_warp_matrix_a.npz').format(self.imageId))
        warp_matrix_m = np.load((data_dir+'/utils/image_alignment/{}_warp_matrix_m.npz').format(self.imageId))

        ima = affine_transform(ima, warp_matrix_a, [nx, ny])
        imm = affine_transform(imm, warp_matrix_m, [nx, ny])

        im = np.concatenate((im3, ima, imm), axis=-1)

        return im


    def create_label(self):

        if self.image is None:
            self.load_image()
        labels = np.zeros(np.append(self.image_size, len(CLASSES)), np.uint8)
        for cl in CLASSES:
            polygon_list = get_polygon_list(self.imageId, cl)
            perim_list, inter_list = genetate_contours(polygon_list, self.image_size, self._xymax)
            mask = generate_mask_from_contours(self.image_size, perim_list, inter_list, cl)
            labels[:, :, cl-1]= mask
        self.label = labels


    def visualize_images(self, plot_all=True):

        if self.label is None:
            self.create_label()

        if plot_all is False:
            fig, axarr = plt.subplots(figsize=[10,10])
            ax = axarr
        elif plot_all is True:
            fig, axarr = plt.subplots(figsize=[20, 20], ncols=3, nrows=3)
            ax = axarr[0][0]

        polygon_list = {}
        for cl in CLASSES:
            polygon_list[cl] = get_polygon_list(self.imageId, cl)
        for cl in CLASSES:
            print '{}: {} \tcount = {}'.format(cl, CLASSES[cl], len(polygon_list[cl]))

        legend = plot_polygon(polygon_list=polygon_list, ax=ax)

        ax.set_xlim(0, self._xymax[0])
        ax.set_ylim(self._xymax[1], 0)
        ax.set_xlabel(self.image_size[0])
        ax.set_ylabel(self.image_size[1])

        if plot_all is True:
            three_band_rescale = scale_percentile(self.three_band_image)
            sixteen_band_rescale = scale_percentile(self.sixteen_band_image)
            plot_image(three_band_rescale, axarr[0][1], self.imageId, '3')
            plot_overlay(three_band_rescale, axarr[0][2], self.imageId, '3', polygon_list, scaler=self.image_size/np.array([-self._xymax[1], -self._xymax[0]]))
            axarr[0][2].set_ylim(self.image_size[0], 0)
            axarr[0][2].set_xlim(0, self.image_size[1])
            plot_image(sixteen_band_rescale, axarr[1][0], self.imageId, 'A', selected_channel=[0, 3, 6])
            plot_image(sixteen_band_rescale, axarr[1][1], self.imageId, 'A', selected_channel=[1, 4, 7])
            plot_image(sixteen_band_rescale, axarr[1][2], self.imageId, 'A', selected_channel=[2, 5, 0])
            plot_image(sixteen_band_rescale, axarr[2][0], self.imageId, 'M', selected_channel=[8, 11, 14])
            plot_image(sixteen_band_rescale, axarr[2][1], self.imageId, 'M', selected_channel=[9, 12, 15])
            plot_image(sixteen_band_rescale, axarr[2][2], self.imageId, 'M', selected_channel=[10, 13, 8])

        ax.legend(handles=legend,
                  bbox_to_anchor=(0.9, 0.95),
                  bbox_transform=plt.gcf().transFigure,
                  ncol=5,
                  fontsize='large',
                  title='Objects-'+self.imageId,
                  framealpha=0.3)


    def apply_crop(self, patch_size, ref_point=[0,0],  method='random'):

        if self.image is None:
            self.load_image()

        crop_area = np.zeros([2,2])
        width = self.image_size[0]
        height = self.image_size[1]

        assert patch_size > 0
        assert patch_size <= width and patch_size <= height

        if method == 'random':
            ref_point[0] = random.randint(0, width - patch_size)
            ref_point[1] = random.randint(0, height - patch_size)
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        elif method == 'grid':
            assert width > ref_point[0] + patch_size
            assert  height > ref_point[1] + patch_size
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        else:
            raise NotImplementedError('"method" should either be "random" or "grid"')
        self.crop_image = crop(self.image, crop_area)



'''
Some nice code:

https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras

def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask
    
def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
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

def make_submit():
    print "make submission file"
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print df.head()
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print idx
    print df.head()
    df.to_csv('subm/1.csv', index=False)
    
    
https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly

dumped_prediction = shapely.wkt.dumps(scaled_pred_polygons)
print('Prediction size: {:,} bytes'.format(len(dumped_prediction)))
final_polygons = shapely.wkt.loads(dumped_prediction)
print('Final jaccard',
      final_polygons.intersection(train_polygons).area /
      final_polygons.union(train_polygons).area)



from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
def mask_to_polygons(mask, epsilon=10., min_area=10.):
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
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


'''

'''
https://www.kaggle.com/resolut/waterway-0-095-lb
this  index can be used to classify water
def CCCI_index(m, rgb):
    RE  = resize(m[5,:,:], (rgb.shape[0], rgb.shape[1]))
    MIR = resize(m[7,:,:], (rgb.shape[0], rgb.shape[1]))
    R = rgb[:,:,0]
    # canopy chloropyll content index
    CCCI = (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)
    return CCCI
# The following is almost enough for classifying water
binary = (CCCI > 0.11).astype(np.float32)
'''