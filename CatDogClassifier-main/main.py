import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
from tenserflow.keras.preprocessing.image import ImageDataGenerator

train_set=r'C:\Users\abaluni\Desktop\test_set\test_set'
test_set=r'C:\Users\abaluni\Desktop\training_set\training_set'


train_gen=ImageDataGenerator(
	rescale=1./255,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
	)

test_gen=ImageDataGenerator(rescale=1./255)



train_generator=train_gen.flow_from_directory(
	train_set,
	target_size=(150,150),
	batch_size=30,
	class_mode='binary'
	)

validation_generator=test_gen.flow_from_directory(
	test_set,
	target_size=(150,150),
	batch_size=30,
	class_mode='binary'
	)


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_state=(150,150,3)))
model.add(layers.MaxPooling2D(((2,2))))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#model.summary()


model.compile(
	loss='binary_crossentropy',
	optimizer=tf.keras.optimizers.RMSporp(lr=1e-4),
	matrics=['acc']
	)


model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=150,
	validation_data=validation_generator,
	validation_steps=20
	)

mdoel.save("CatDogDistinguisher.model")
    pd.DataFrame(model.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()