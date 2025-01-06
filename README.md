# Boosting multi-demographic federated learning for chest X-ray diagnosis using general-purpose self-supervised representations


Overview
------

...

### Prerequisites

The software is developed in **Python 3.9**. For the deep learning, the **PyTorch 2.0** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate FLTLCXR
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for federated learning, self-supervised transfer learning, training and evaluation of the networks, statistical analysis, data augmentation, image analysis, and pre-processing are available here.

1. Everything can be run from *./main_fltl.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, augmentation, and loading files.
* *./FL/* directory contains all the FL processes.
* *./Train_Valid_fltl.py* contains the training and validation processes.
* *./Prediction_fltl.py* all the prediction and testing processes.

------
### In case you use this repository, please cite the original paper
