__author__ = 'rj'


'''
Purposes:
 1. data visualization
 2. evaluation of data augamentation
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

data_dir = './data'

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



_df = pd.read_csv(data_dir+'/train_wkt_v4.csv', names=['ImageId', 'ClassId', 'MultipolygonWKT'], skiprows=1)
_df1 = pd.read_csv(data_dir+'/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
#duplicates = ['6110_1_2', '6110_3_1']
duplicates = []

train_wkt_v4 = _df[np.invert(np.in1d(_df.ImageId, duplicates))]
grid_sizes = _df1[np.invert(np.in1d(_df1.ImageId, duplicates))]

all_train_names = train_wkt_v4.ImageId.unique()

image_IDs_dict = dict(zip(np.arange(all_train_names.size),all_train_names))



def rescale(im, shape_out):
    '''
    
    :param im: 
    :param shape_out: 
    :return: rescaled image
    '''
    return cv2.resize(im, (shape_out[0], shape_out[1]), interpolation=cv2.INTER_CUBIC)



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



def rotate():
    return 0



def crop():
    return 0



def class_label_stat():
    return 0



class ImageData():

    def __init__(self, image_id):

        self.imageId = image_IDs_dict[image_id]
        self.three_band_image = None
        self.sixteen_band_image = None
        self.image = None
        self.image_size = None
        self._xymax = None
        self.label = None


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
            '3': '{}/three_band/{}.tif'.format(data_dir, self.imageId),
            'A': '{}/sixteen_band/{}_A.tif'.format(data_dir, self.imageId),
            'M': '{}/sixteen_band/{}_M.tif'.format(data_dir, self.imageId),
            'P': '{}/sixteen_band/{}_P.tif'.format(data_dir, self.imageId),
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

        ima = rescale(ima, [nx, ny])
        imm = rescale(imm, [nx, ny])

        warp_matrix_a = np.load('utils/image_alignment/{}_warp_matrix_a.npz'.format(self.imageId))
        warp_matrix_m = np.load('utils/image_alignment/{}_warp_matrix_m.npz'.format(self.imageId))

        ima = affine_transform(ima, warp_matrix_a, [nx, ny])
        imm = affine_transform(imm, warp_matrix_m, [nx, ny])

        im = np.concatenate((im3, ima, imm), axis=-1)

        return im


    def create_label(self):

        if self.image is None:
            self.load_image()
        labels = np.zeros(self.image_size, np.uint8)
        for cl in CLASSES:
            polygon_list = get_polygon_list(self.imageId, cl)
            perim_list, inter_list = genetate_contours(polygon_list, self.image_size, self._xymax)
            mask = generate_mask_from_contours(self.image_size, perim_list, inter_list, cl)
            labels += mask
        self.label = labels