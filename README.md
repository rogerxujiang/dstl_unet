# A U-net neural network for objection detection (or segmentation) of satellite images
Dstl Satellite Imagery Feature Detection
# Prerequisites

# Download the data

Download the data from kaggle website: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data
Put the data into the `./data/` folder.

# Train the model
The model is built to train a voxel-wise binary classifier for each of the 10 classes. Change the parameter `class_type` to a number of `0-9` in `./hypes/hypes.json` to switch between classes. Run the following in the terminal to train the model:
```
python train.py |& tee output.txt
```
All the print out will be stored in `output.txt`
# Visualize the training
```
tensorboard --port 6006 --logdir summary_path --host 127.0.0.1
```
![Learning curve of training](https://user-images.githubusercontent.com/6231739/29622323-1a8557e2-87f1-11e7-9110-96f7a9a2a4ef.png)
![Learning curve of validation](https://user-images.githubusercontent.com/6231739/29622328-1d16a7cc-87f1-11e7-8137-4cd07c1d9af7.png)
# Make predictions
```
python inference.py |& tee test_output.txt
```
All the print out will be stored in `test_output.txt`
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
