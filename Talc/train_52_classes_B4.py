#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append(".")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
import matplotlib.pylab as plt
from pathlib import Path
from tensorflow.keras import layers

#from PlayingCardsGenerator import CardsDataGenerator
import datetime


model_name_it = "/home/drew.burritt/enel645/term-project/Outputs/Efficient_net_B4_it_52.h5"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
else:
  print("No GPU device found")


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)
monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it, monitor='val_loss',                                             verbose=1,save_best_only=True,                                             save_weights_only=False,                                             mode='min')

def scheduler(epoch, lr):
    if epoch%30 == 0 and epoch!= 0:
        lr = lr/2
    return lr


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

gen_params = {"featurewise_center":False,              "samplewise_center":False,              "featurewise_std_normalization":False,              "samplewise_std_normalization":False,              "rotation_range":90,              "width_shift_range":0.3,              "height_shift_range":0.3,               "shear_range":0.3,               "zoom_range":0.3,              "vertical_flip":True,               "brightness_range": (0.2, 2)}


generator = ImageDataGenerator(**gen_params, validation_split=0.2,  preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)


bs = 16 # batch size

path = Path("/home/drew.burritt/enel645/term-project/dataset/")

img_height = 380
img_width = 380

classes_names = ["2_clubs","2_diamonds","2_hearts","2_spades",               "3_clubs","3_diamonds","3_hearts","3_spades",               "4_clubs","4_diamonds","4_hearts","4_spades",               "5_clubs","5_diamonds","5_hearts","5_spades",               "6_clubs","6_diamonds","6_hearts","6_spades",               "7_clubs","7_diamonds","7_hearts","7_spades",               "8_clubs","8_diamonds","8_hearts","8_spades",               "9_clubs","9_diamonds","9_hearts","9_spades",               "10_clubs","10_diamonds","10_hearts","10_spades",               "ace_clubs","ace_diamonds","ace_hearts","ace_spades",               "jack_clubs","jack_diamonds","jack_hearts","jack_spades",               "king_clubs","king_diamonds","king_hearts","king_spades",               "queen_clubs","queen_diamonds","queen_hearts","queen_spades"]

train_generator = generator.flow_from_directory(
    directory = path,
    target_size=(img_height, img_width),
    batch_size=bs,
    class_mode="categorical",
    subset='training',
    shuffle = True,
    interpolation="nearest",
    seed=42,
    classes=classes_names) # set as training data

validation_generator = generator.flow_from_directory(
    directory = path,
    target_size=(img_height, img_width),
    batch_size=bs,
    class_mode="categorical",
    subset='validation',
    interpolation="nearest",
    seed=42,
    classes=classes_names) # set as validation data

# Defining the model

trainable_flag = True
include_top_flag = False
weigths_value = 'imagenet'

if trainable_flag:
    include_top_flag = True
    weigths_value = None
else:
    include_top_flag = False
    weigths_value = 'imagenet'    

print(weigths_value)
print(include_top_flag)
print(trainable_flag)


inputs = layers.Input(shape=(img_height,img_width,3))
outputs = tf.keras.applications.EfficientNetB4(include_top=include_top_flag, weights=weigths_value,drop_connect_rate=0.3, classes=len(classes_names))(inputs)
model = tf.keras.Model( inputs,  outputs)

print("Initial Training Model")
print(model.summary())


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3), #
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#model = tf.keras.models.load_model(model_name_it)
#model.summary()


history_it = model.fit(train_generator, epochs=1000, verbose = 1,                        workers=8, validation_data = (validation_generator),  callbacks= [monitor_it,early_stop,lr_schedule])



model.save('/home/drew.burritt/enel645/term-project/Outputs/final_it_EfficientNetB5_52_last_model.h5')
np.save('/home/drew.burritt/enel645/term-project/Outputs/efficientNetB4_history.npy',history_it.history)


#model = tf.keras.models.load_model('/home/drew.burritt/enel645/term-project/Outputs\Efficient_net_B0_it_52.h5')
#model.save('final_it_EfficientNetB0_52_96_percent_v2_best_model.h5')



