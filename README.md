(Check out my Medium post (https://goo.gl/Ussdr1) for more details of this project.)

# A U-net based on Tensorflow for objection detection (or segmentation) of satellite images

The goal of this project is to develop models for Dstl Satellite Imagery Feature Detection contest on kaggle. The result scores 0.46 on the public test data set and 0.44 on the private test data set, would rank No. 7 out of 419 on the private leaderboard. 

The training dataset includes 25 images, each with 20 channels (RGB band (3 channels)  + A band (8 channels) + M band (8 channels) + P band (1 channel)), and the corresponding labels of objects. There are 10 types of overlapping objects labeled with contours (`wkt` type of data), including 0. Buildings, 1. Misc, 2. Road, 3. Track, 4. Trees, 5. Crops, 6. Waterway, 7. Standing water, 8. Vehicle Large, 9. Vehicle Small.

This code converts the contours into masks, and then trains a pixel-wise binary classifier for each class of object. A U-net with batch norm developed in tensorflow is used as the classification model. A combination of cross entropy and soft Jaccard index, and the Adam optimizer are the loss function and the optimizer respectively. The following figures show examples of the training features and labels from one of the training examples. The code to generate these figures can be found in `visualization.ipynb`.

![All bands](https://user-images.githubusercontent.com/6231739/29629077-8f59cdc8-8805-11e7-92d1-978bfc3b2f6d.png)
![Labels](https://user-images.githubusercontent.com/6231739/29629079-915f1f4c-8805-11e7-9c02-02e1c40500f7.png)

This figure shows the statistics of percentage area for all classes of all the training data. (Note: on some images, the sum is over 100% because of overlap between classes.)
![Stats](https://user-images.githubusercontent.com/6231739/29629084-94624fde-8805-11e7-913b-f852ec4d79f8.png)

# Prerequisites
## Suggested hardware
* Nvidia K80 Tesla GPU
* 61 GB RAM

The model was developed and trained on a `p2.xlarge` instance on AWS, which comes with the above hardware. At the beginning of the training for each class, all the 25 training images and the corresponding labels are loaded into RAM to avoid file  I/O during the training, which can slow down the training. Therefore a large RAM (up to 50 GB) is required. The batch size and patches size of images in training and predictions are also customized for the ~11 GB memory on K80 GPU. These parameters should be adjusted according to your hardware.

## Software and Packages
* python == 2.7
* tensorflow == 1.0.1
* descartes == 1.1.0
* matplotlib == 2.0.0
* numpy == 1.12.0
* opencv-python == 3.3.0.9
* pandas == 0.20.3
* seaborn == 0.7.1
* shapely == 1.6.0
* simplejson == 3.10.0
* tifffile == 0.12.0

To install all the requirements:
```
pip install -r requirements.txt
conda install -c https://conda.binstar.org/menpo opencv3
```


# Download the data

Download the data from contest website: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data

Put the data into the `./data/` folder.

# Train the model
The model is built to train a voxel-wise binary classifier for each of the 10 classes. Change the parameter `class_type` to a number of `0-9` in `./hypes/hypes.json` to switch between classes. Run the following in the terminal to train a model for each class:
```
python train.py |& tee output.txt
```
All the print out is saved in `output.txt`. All other logs for each training is saved at a folder in `./log_dir`, with a folder name of `./log_dir/month-day_hour-min_lossfunction`, including a TF checkpoint for every 1000-batch, a summary point for every 100-batch, and the hyper parameters for the training. The last TF checkpoint is used to generate predictions.
The final version of this code includes all the labeled data for training. You can set the `test_names` in `./utils/train_utils.py`, and exclude them from the `train_names` parameters to perform cross validation.

# Visualize the training
To monitor the training on the fly using `tensorboard`, run the following code in terminal:
```
tensorboard --port 6006 --logdir summary_path --host 127.0.0.1
```
The following figures are examples of learning curves for the training of class 0, Bldg.
![Learning curve of training](https://user-images.githubusercontent.com/6231739/29622323-1a8557e2-87f1-11e7-9110-96f7a9a2a4ef.png)
![Learning curve of validation](https://user-images.githubusercontent.com/6231739/29622328-1d16a7cc-87f1-11e7-8137-4cd07c1d9af7.png)

# Make predictions
Modify the `save_path` parameter of `saver.restore()` in `inference.py` to the path of the last checkpoint and change the `class_type` in `./hypes/hypes.json` to the desired class type to generate predictions:
```
python inference.py |& tee test_output.txt
```
All the print out will be saved in `test_output.txt`. The predictions will be saved in a CSV file `./submission/class_{class_type}.csv`.

# Merge submission and submit
To merge the prediction files of all classes (e.g. `./submission/class_0.csv` for class 0), run the following in terminal:
```
python merge_submission.py
```
A few errors of `non-noded intersection` were encountered during my submission. This can be fixed by running `python topology_exception.py` for each of the error. The script `topology_exception.py` will create a hole around the `point`, which can be found from the error message. You could also run the following in a python console:

```
repair_topology_exception('submission/valid_submission.csv', 
                           precision=6, 
                           image_id='6100_0_2',
                           n_class=4,
                           point= (0.0073326816112855523, -0.0069418340919529765),
                           side=1e-4)
```
# Results
The online evaluation returns a score of 0.44, which would rank No. 7 on the private leaderboard. The following figures show the comparison between the true label and predicted label.
![Comparison](https://user-images.githubusercontent.com/6231739/29648659-4b9e8526-885d-11e7-90f3-43a602c1277a.png)

# Remarks
1. The model was primarily developed based on performance on class 0 (Bldg). It can be further improved for other classes, by customizing model parameters for each class.
2. Check out [my medium post](https://medium.com/@rogerxujiang/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40) for more details about this project.
