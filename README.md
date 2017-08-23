# A U-net using Tensorflow for objection detection (or segmentation) of satellite images
Dstl Satellite Imagery Feature Detection
# Prerequisites

## Hardware
* Nvidia K80 Tesla GPU
* 61 GB ram
The author developed and trained the model on a `p2.xlarge` instance on AWS, which comes with the above hardware. At the beginning of the training, all the training images and labels are loaded into RAM to avoid the slow file  I/O, so a large RAM (up to 50 GB) is required. The batch size and size of patches of images in `train.py` and `inference.py` are customized for the ~11 GB memory on K80 GPU.

## Software and Packages
* python == 2.7
* descartes == 1.1.0
* matplotlib == 2.0.0
* numpy == 1.12.0
* opencv-python == 3.3.0.9
* pandas == 0.20.3
* seaborn == 0.7.1
* shapely == 1.6.0
* simplejson == 3.10.0
* tensorflow == 1.0.1
* tifffile == 0.12.0

To install all the requirements:
```
pip install -r requirements.txt
conda install -c https://conda.binstar.org/menpo opencv3
```


# Download the data

Download the data from kaggle website: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data
Put the data into the `./data/` folder.

# Train the model
The model is built to train a voxel-wise binary classifier for each of the 10 classes. Change the parameter `class_type` to a number of `0-9` in `./hypes/hypes.json` to switch between classes. Run the following in the terminal to train the model:
```
python train.py |& tee output.txt
```
All the print out is saved in `output.txt`. All other log for each training is saved at a folder in `./log_dir`, with a folder name of `./log_dir/month-day_hour-min_lossfunction`, including a TF checkpoint for every 1000-batch, a summary point for every 100-batch, and the hyper parameter used for the training. The last TF checkpoint is used to generate predictions.
The final version of this code use all the labeled data for training. You can set the `test_names` in `./utils/train_utils.py`, and exclude them from `train_names` for cross validation.

# Visualize the training
To see the learning curve, run the following code in terminal:
```
tensorboard --port 6006 --logdir summary_path --host 127.0.0.1
```
The following figures are examples of learning curves for the training of class 0, Bldg.
![Learning curve of training](https://user-images.githubusercontent.com/6231739/29622323-1a8557e2-87f1-11e7-9110-96f7a9a2a4ef.png)
![Learning curve of validation](https://user-images.githubusercontent.com/6231739/29622328-1d16a7cc-87f1-11e7-8137-4cd07c1d9af7.png)

# Make predictions
```
python inference.py |& tee test_output.txt
```
All the print out will be saved in `test_output.txt`
# Merge submission and submit
To merge the prediction files for all classes (e.g. `./submission/class_0.csv` for class 0), run the following in terminal:
```
python merge_submission.py
```
A few errors of `non-noded intersection` were encountered during my submission. This can be fixed by running `python topology_exception.py` for each of the error. The script `topology_exception.py` will create a hole around the `point` parameter, which can be found from the error message. You could also run the following in a python console:

```
repair_topology_exception('submission/valid_submission.csv', 
                           precision=6, 
                           image_id='6100_0_2',
                           n_class=4,
                           point= (0.0073326816112855523, -0.0069418340919529765),
                           side=1e-4)
```
# Remarks
