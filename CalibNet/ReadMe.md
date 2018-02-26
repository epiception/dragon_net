CalibNet
==========
---
## Instructions

First run `parser_create_2_color.py`. Make sure the dataset folder contains all raw _sync folders.

    $ python parser_create_2_color.py /path/to/main/dataset/folder

Download ResNet parameters (as a .json file) from this [link](https://drive.google.com/open?id=1nKiT4KZV6YOcFquRGRr0F1XBUNhquSW1), and set this path to the json file in `config_res.py`

Briefly run `resnet_model_full.py` to check for any errors related to tf_ops (for Earth Mover's distance).
    
    $ CUDA_VISIBSLE_DEVICES=<gpu-ids> python -B resnet_model_full.py
    
If no errors, then training should begin without any issues. All tunable hyperparameters can be found in `config_res.py`. 
Tuning is required mainly for `alpha_const`, `beta_const`, `learning_rate` (occasionally). Set `current_epoch`, to load the checkpoint, by number.

#### pip install requirements
 
* [tqdm](https://pypi.python.org/pypi/tqdm)
* [glob](https://docs.python.org/2/library/glob.html)
* matplotlib (for pyplot)
* scipy.misc
