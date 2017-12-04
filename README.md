# Bittiger_CapstoneWork

## Group menbers
Wei Luo, Meng Zhang and Renyuan Zhang

## Process
We train our model based on the given structure of YOLO v2. In our tests, we've tried to turn some parameters or settings to find a better result. The learning rate, resnet model type and learning rate scheduler are turned in our model. For our last model some key features are illustrated in the following sections.

### Last training epoch 
Here are some result of our model. As the result, the accuracy, precision and recall are extremely high, but loss is almost unchanged during the training procedure.  

```text
Epoch[799] Validation-c_accuracy=1.000000
Epoch[799] Validation-c_precision=1.000000
Epoch[799] Validation-c_recall=1.000000
Epoch[799] Validation-c_diff=0.009402
Epoch[799] Validation-x_diff=0.098163
Epoch[799] Validation-y_diff=0.075695
Epoch[799] Validation-w_diff=0.154178
Epoch[799] Validation-h_diff=0.198570
Epoch[799] Validation-loss=0.762392
Epoch[799] Validation-cls_diff=0.000706
```

## Files in github
1. ``run_train.py``: the main file of training
2. ``data_ulti.py``: handle the data
3. ``testandprint.py``: randomly choose 30 images in testing data set and show the result after model. The test images with result are stored in ``./result`` directory
4. ``toJson.py``: according to the request of this work, we need to create json file to store last test result. This python file will create the test result based on testing data and store in ``result.json``. 

> Note: the training and testing data given by Bittiger will not be provided in this github.
> 

## Files to download
Several files are too big to download directly from github. Here is a link:https://pan.baidu.com/s/1c2GXQP6 with PIN:6of9
1. files in ``./log``: log files for tensorboard. 
2. files in ``./pretrained_models``: some pretrained models given by Bittiger
3. trained final model: ``drive_full_detect-symbol.json`` and ``drive_full_detect-800.params``