# Training Process for DR Detection

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.models import save_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


train_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\FDATASET\train'

valid_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\FDATASET\validation'


IMG_SHAPE = 224
batch_size = 32

image_gen_train = ImageDataGenerator()
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=False,
                                                     target_size=(
                                                         IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

image_generator_validation = ImageDataGenerator()
val_data_gen = image_generator_validation.flow_from_directory(batch_size=batch_size,
                                                              directory=valid_dir,
                                                              shuffle=False,
                                                              target_size=(
                                                                  IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')

pre_trained_model = tf.keras.applications.InceptionV3(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in pre_trained_model.layers:
    # print(layer.name)
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed10')
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(pre_trained_model.input, x)
# print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam', metrics=['acc'])

training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(val_data_gen.samples / batch_size)

history = model.fit(train_data_gen,
                    steps_per_epoch=training_steps_per_epoch,
                    epochs=40,
                    validation_data=val_data_gen,
                    validation_steps=validation_steps_per_epoch,
                    batch_size=batch_size,
                    verbose=1)
print('Training Completed!')

model.save(r'C:\Users\Computer\Desktop\project-docs\DR app\Model1a.h5')

# summarize history for accuracy

history_dict = history.history
print(history_dict.keys())
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

fig1 = plt.figure(1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(False)
plt.show()


# summarize history for loss
fig2 = plt.figure(2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(False)
plt.show()


print('completed')
exit()
