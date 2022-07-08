import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading dataset


fashion_minst=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion_minst.load_data()
class_name=["Tshirt","Trouser","Full-Slieves","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Boot"]
print(class_name[y_train[1]])

#normalizaion


x_train=x_train/255.
y_test=x_test/255.

#model creation


# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
# model.add(tf.keras.layers.Dense(300,activation="relu"))
# model.add(tf.keras.layers.Dense(100,activation="relu"))
# model.add(tf.keras.layers.Dense(10,activation="softmax"))
# model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=80)
# model.save("FashionClassifier")


#model calling
model=tf.keras.models.load_model("FashionClassifier")

#prediction

y=model.predict(x_test)
print(np.argmax(y))
