#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anelmusic
"""

import tensorflow as tf
import cv2 as cv2
import albumentations as A

"##############################################################################"
# Provide path to dataset and label-file
DATA_PATH = "/home/anelmusic/anel_projects/datasets/camvid/CamVid/"
CLASS_CSV_FILENAME = "class_dict.csv"

"##############################################################################"
# Provide path to pretrained encoder weights
VGG_WEIGHTS_PATH = "/home/anelmusic/anel_projects/camvid/camvid_semantic_segmentation_tf2/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
"##############################################################################"
# Define basic hyperparams
BATCH_SIZE = 32
NUM_PIXELS_SQRT  = 256
IM_SIZE = (NUM_PIXELS_SQRT, NUM_PIXELS_SQRT)
OUTPUT_CHANNELS = 32
NUM_EPOCHS = 100
NUM_TRAIN_IMGS = 369
NUM_VAL_IMGS = 100

"##############################################################################"
# Define augmentation for: image and mask / image only
IMAGE_MASK_AUGMENTATIONS = A.Compose([A.Rotate(limit=20),
                                      A.HorizontalFlip(p=0.3),
                                      A.Resize(NUM_PIXELS_SQRT, NUM_PIXELS_SQRT, interpolation= cv2.INTER_NEAREST, p = 1),
                                      A.RandomSizedCrop(min_max_height=(int(NUM_PIXELS_SQRT*0.5), int(NUM_PIXELS_SQRT*1)), 
                                                        height=NUM_PIXELS_SQRT, width=NUM_PIXELS_SQRT, p=0.8),
                                      ])
IMAGE_AUGMENTATION =  A.Compose([A.RandomGamma(p=0.8)])
AUGMENTATIONS = {"img_augmentation": IMAGE_AUGMENTATION, "img_mask_augmentation": IMAGE_MASK_AUGMENTATIONS}

"##############################################################################"
# Define optimizer loss and metrics
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1E-3, momentum=0.9, nesterov=True)
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]

"##############################################################################"
# Define mode: training / inference
MODE = "training" 

"##############################################################################"
# Define where to store or from where to load weights after training
# In training mode this is the path for storing the weights
# In inference mode this is the path for loading the weights
SAVE_LOAD_WEIGHTS_PATH = '/home/anelmusic/anel_projects/camvid/camvid_semantic_segmentation_tf2/checkpoints/FCN8_SEGMENTATION_WEIGHTS'
