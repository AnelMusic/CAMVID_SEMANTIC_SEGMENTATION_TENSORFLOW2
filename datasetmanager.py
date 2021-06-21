#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: anelmusic
"""

import os
from glob import glob

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

import tensorflow as tf

class DatasetManager:
    def __init__(self, dataset_path, class_csv_file, bs = 10, augmentations = None):
        self.dataset_path = dataset_path
        self.class_csv_file = class_csv_file
        self._class_df = None
        self.label_dict = None
        self.batchsize = bs
        self.seed = 1
        self.dataset_dirname  = None
        self.dataset_raw = None
        self.buffer_size = 10000
        self.image_size = None
        self.dataset_img_filenames = None 
        self.transforms_image = None
        self.transforms_image_and_mask = None
        self.augmentations = augmentations
        self._init_manager()

    def _init_manager(self):
        self._prepare_labeldict()
        if self.augmentations != None:
            self.transforms_image = self.augmentations["img_augmentation"]
            self.transforms_image_and_mask = self.augmentations["img_mask_augmentation"]

    def _prepare_labeldict(self):
        self._class_df = pd.read_csv(os.path.join(self.dataset_path, self.class_csv_file))
        colors = self._class_df[["r", "g", "b"]].values.tolist()
        colors = [tuple(color) for color in colors]
        category = self._class_df[["name"]].values.tolist()
        self.label_dict = {"COLORS": colors, "CATEGORIES": category}
    
    def show_label_encoding(self):
        return self._class_df
    
    def get_label_info(self):
        return self.label_dict
    
    def _augment_data(self, datapoint): 
        input_image = datapoint['image']
        input_mask = datapoint['segmentation_mask']
        return input_image, input_mask

    def _process_data(self, image_path, mask_path):
        image, mask = self._get_image(image_path), self._get_image(mask_path, mask=True)
        if self.dataset_dirname == "train":
            aug_img = tf.numpy_function(func=self._aug_training, inp=[image, mask], Tout=(tf.float32,tf.float32))
            datapoint = self._normalize_img_and_colorcorrect_mask(aug_img[0],aug_img[1])
            return datapoint[0], datapoint[1]
        else:
            aug_img = tf.numpy_function(func=self._aug_basic, inp=[image, mask], Tout=(tf.float32,tf.float32))
            datapoint = self._normalize_img_and_colorcorrect_mask(aug_img[0],aug_img[1])
            return datapoint[0], datapoint[1]

    def _get_filenpaths(self): 
        dataset_img_filenames = tf.data.Dataset.list_files(self.dataset_path + self.dataset_dirname+"/"+ "*.png", seed=self.seed)
        image_paths = os.path.join(self.dataset_path,self.dataset_dirname, "*")
        mask_paths = os.path.join(self.dataset_path,self.dataset_dirname+"_labels", "*")
        image_list = sorted(glob(image_paths))
        mask_list = sorted(glob(mask_paths))
        return image_list, mask_list     

    def _get_image(self, image_path,  mask=False):
        img = tf.io.read_file(image_path)
        if not mask:
            img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        else:
            img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        return img

    def _aug_training(self,image, mask):
        # augment image and mask
        img_mask_data = {"image":image, "mask":mask}
        aug_image_and_mask = self.transforms_image_and_mask(**img_mask_data)
        aug_img = aug_image_and_mask["image"]
        aug_mask = aug_image_and_mask["mask"]
        # augment image only
        img_data = {"image":aug_img}
        aug_data =  self.transforms_image(**img_data)
        aug_img = aug_data["image"]

        aug_img = tf.cast(aug_img, tf.float32)
        aug_img = tf.image.resize(aug_img, size=self.image_size)
        aug_mask = tf.cast(aug_mask, tf.float32)
        aug_mask = tf.image.resize(aug_mask, size=self.image_size)
        aug_img = tf.clip_by_value(aug_img, 0,255)
        return (aug_img, aug_mask)
    
    def _aug_basic(self,image, mask):
        aug_img = tf.cast(image, tf.float32)
        aug_img = tf.image.resize(aug_img, size=self.image_size)
        aug_mask = tf.cast(mask, tf.float32)
        aug_mask = tf.image.resize(aug_mask, size=self.image_size)
        return (aug_img, aug_mask)

    def _normalize_img_and_colorcorrect_mask(self,input_image, input_mask): 
        input_image = tf.cast(input_image, tf.float32) / 255.0
        one_hot_map = []
        for color in self.label_dict["COLORS"]:
            class_map = tf.reduce_all(tf.equal(input_mask, color), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        return (input_image, one_hot_map)
        
    @tf.function    
    def _set_shapes(self, img, mask):  
        img.set_shape((self.image_size[0],self.image_size[1],3))
        mask.set_shape((self.image_size[0],self.image_size[1],32))
        return img,mask
    
    def _restore_original_mask_colors(self, mask):
        new_mask = mask
        h,w = new_mask.shape 
        new_mask = np.reshape(new_mask, (h*w,1))
        dummy_mask = np.ndarray(shape=(h,w, 3))
        dummy_mask =  np.reshape(dummy_mask, (h*w, 3))
        for idx, pixel in enumerate(new_mask):
            dummy_mask[idx] = np.asarray(self.label_dict["COLORS"][int(pixel)])
        return np.reshape(dummy_mask, (h,w,3))/255.
    
    def _get_prepared_dataset(self):
        if self.dataset_dirname == "train":
            self.dataset = self.dataset.map(self._process_data, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            self.dataset = self.dataset.map(self._set_shapes, num_parallel_calls=self.AUTOTUNE).shuffle(150).repeat().batch(self.batchsize ).prefetch(self.AUTOTUNE)
            self.train_ds = self.dataset
            return self.train_ds
        elif self.dataset_dirname == "val":
            self.dataset = self.dataset.map(self._process_data, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            self.dataset = self.dataset.map(self._set_shapes, num_parallel_calls=self.AUTOTUNE).repeat().batch(self.batchsize ).prefetch(self.AUTOTUNE)
            self.val_ds = self.dataset
            return self.val_ds
        elif self.dataset_dirname == "test":
            self.dataset = self.dataset.map(self._process_data, num_parallel_calls=self.AUTOTUNE).prefetch(self.AUTOTUNE)
            self.dataset = self.dataset.map(self._set_shapes, num_parallel_calls=self.AUTOTUNE).repeat().batch(1).prefetch(self.AUTOTUNE)
            self.test_ds = self.dataset
            return self.test_ds

    def show_batch(self, ds, fsize = (15,5)):
        image_batch, label_batch = next(iter(ds)) 
        image_batch = image_batch.numpy()
        label_batch = label_batch.numpy()
        for i in range(len(image_batch)):
            fig, (ax1, ax2 )= plt.subplots(1, 2, figsize=fsize)
            fig.suptitle('Image Label')
            ax1.imshow(image_batch[i])
            ax2.imshow(self._restore_original_mask_colors(np.argmax(label_batch[i], axis=-1)))

    def get_dataset(self, dataset_dirname, image_size = (128,128)): 
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.image_size = image_size 
        self.dataset_dirname = dataset_dirname
        self.image_list, self.mask_list  = self._get_filenpaths()
        self.dataset = tf.data.Dataset.from_tensor_slices((self.image_list, self.mask_list))
        return self._get_prepared_dataset()