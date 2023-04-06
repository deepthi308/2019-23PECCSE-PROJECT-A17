# Training Process for DR Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import Xception
from keras.models import load_model

num_classes = 3
IMAGE_SHAPE = [224, 224]
batch_size = 32  # change for better accuracy based on your dataset
epochs = 100  # change for better accuracy based on your dataset

train_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\SDATASET\train'
valid_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\SDATASET\validation'

EF = Xception(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
for layer in EF.layers:
    layer.trainable = False
x = Flatten()(EF.output)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=EF.input, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['acc'])

trdata = ImageDataGenerator()
train_data_gen = trdata.flow_from_directory(directory=train_dir, target_size=(
    224, 224), shuffle=False, class_mode='categorical')
valdata = ImageDataGenerator()
val_data_gen = valdata.flow_from_directory(directory=valid_dir, target_size=(
    224, 224), shuffle=False, class_mode='categorical')

training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(val_data_gen.samples / batch_size)
history = model.fit(train_data_gen, steps_per_epoch=training_steps_per_epoch,
                    validation_data=val_data_gen, validation_steps=validation_steps_per_epoch, epochs=epochs, verbose=1)
print('Training Completed!')

model.save(r'C:\Users\Computer\Desktop\project-docs\DR app\Model2a.h5')

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
