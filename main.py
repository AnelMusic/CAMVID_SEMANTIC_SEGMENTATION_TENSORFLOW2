#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: anelmusic
"""

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import config 
import datasetmanager
import fcn8

assert tf.test.gpu_device_name(), "Your GPU is not itilized"


def main():
    data_manager = DatasetManager(config.DATA_PATH, config.CLASS_CSV_FILENAME, config.BATCH_SIZE, augmentations = config.AUGMENTATIONS)
    train_ds = data_manager.get_dataset("train",  image_size = config.IM_SIZE) 
    val_ds = data_manager.get_dataset("val", image_size = config.IM_SIZE) 
    test_ds = data_manager.get_dataset("test", image_size = config.IM_SIZE)
    
    if config.MODE == "training":
        # (Optional) Investigate augmentation results
        #data_manager.show_batch(train_ds)
        model = segmentation_model()    
        train_model(model, train_ds, val_ds, config.NUM_EPOCHS, config.BATCH_SIZE, 
                    config.NUM_TRAIN_IMGS, config.NUM_VAL_IMGS, config.OPTIMIZER, 
                    config.METRICS, config.LOSS)
    elif config.MODE == "inference":
        model = segmentation_model()   
        evaluate_model(model, val_ds,config.NUM_EPOCHS, config.BATCH_SIZE, 
                       config.NUM_TRAIN_IMGS, config.NUM_VAL_IMGS, config.OPTIMIZER, 
                       config.METRICS, config.LOSS)
    exit()

if __name__ == "__main__":
    main()
    
    