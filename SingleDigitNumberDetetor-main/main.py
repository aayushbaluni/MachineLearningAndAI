import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# dset=tf.keras.datasets.mnist
# #label data--> to traning data and testing data
# (image_train,class_train),(image_test,class_test) =dset.load_data()
# image_train=tf.keras.utils.normalize(image_train,axis=1)
# image_test=tf.keras.utils.normalize(image_test,axis=1)


# #nn model

# model = tf.keras.models.Sequential()
# #flatten layer --> convert grid to single or in 28*28 of 1 layer
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# #basic nural network layer where each nuron of one layer is connected to onother nuron
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# #output layer for 0-9 digits so 10 neurons , softmax --> insures all 10 nurons add up to one
# model.add(tf.keras.layers.Dense(10,activation='softmax'))
# #compile
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# #train
# model.fit(image_train,class_train,epochs=10)
# #save
# model.save('handwritten.model')

model=tf.keras.models.load_model('handwritten.model')
image_number=1
while os.path.isfile(f"digit/digits{image_number}.png"):
	try:
		img=cv2.imread(f"digit/digits{image_number}.png")[:,:,0]
		img=np.invert(np.array([img]))
		predict=model.predict(img)
		print(f"Pridicted value: {np.argmax(predict)}")
		plt.imshow(img[0],cmap=plt.cm.binary)
		plt.show()
	except:
		print("Error")	
	finally:
		image_number+=1
print("Task finished.")
