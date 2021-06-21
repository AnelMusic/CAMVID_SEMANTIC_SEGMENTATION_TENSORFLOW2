#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:59:18 2021

@author: anelmusic
"""

def train_model(model, train_ds, val_ds, epochs, batch_size, num_train_imgs, num_val_imgs, optimizer, metrics, loss):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    steps_per_epoch =num_train_imgs//batch_size
    val_steps = num_val_imgs//batch_size
    
    model_history = model.fit(train_ds, epochs=epochs,
                              steps_per_epoch=val_steps,
                              validation_steps=val_steps,
                              validation_data=val_ds)   
    model.save_weights(config.SAVE_LOAD_WEIGHTS_PATH)

def evaluate_model(model, ds, epochs, batch_size, num_train_imgs, num_val_imgs, optimizer, metrics, loss):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    steps_per_epoch =num_train_imgs//batch_size
    val_steps = num_val_imgs//batch_size
    model.load_weights(config.SAVE_LOAD_WEIGHTS_PATH)
    loss, acc = model.evaluate(ds, steps= val_steps)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))