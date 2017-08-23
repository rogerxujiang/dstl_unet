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
# Visualize the training
```
tensorboard --port 6006 --logdir your_summary_path
```
# Make predictions
```
python inference.py |& tee test_output.txt
```

# Merge submission and submit
To merge the prediction files for all classes (e.g. `./submission/class_0.csv` for class 0), run the following in terminal:
```
python merge_submission.py
```

```
python topology_exception.py
```

# Remarks
