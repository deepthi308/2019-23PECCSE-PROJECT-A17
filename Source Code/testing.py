##### Testing Process ####

# Read test image

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


img = Image.open(
    r"C:\Users\Computer\Desktop\project-docs\DR app\Test Images\E1.jpg")
img = np.asarray(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# fig1 = plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.grid(False)
plt.show()


# Noise removal using median filter

FI = cv2.medianBlur(img, 3)
# fig2 = plt.figure()
plt.imshow(cv2.cvtColor(FI, cv2.COLOR_BGR2RGB))
plt.title('Noise Removal using Median Filter')
plt.grid(False)
plt.show()

# Load Trained Model

OP1 = load_model(
    r'C:\Users\Computer\Desktop\project-docs\DR app\Model1.h5')

# DR Detection

IMG_SIZE = 224
IM = np.array(FI)
img_array = cv2.cvtColor(IM, cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
N = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(N.shape)

prediction = OP1.predict(N)

class_labels = ['Diabetic Retinopathy', 'Normal']
prediction = np.argmax(prediction, axis=-1)
print(prediction)

print(class_labels[prediction[0]])
label = class_labels[prediction[0]]

font = cv2.FONT_HERSHEY_SIMPLEX
orgin = (550, 300)
fontScale = 10
color = (0, 255, 0)
thickness = 10
image = cv2.putText(IM, label, orgin, font,
                    fontScale, color, thickness, cv2.LINE_AA)

# fig4 = plt.figure()
plt.grid(False)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('DR Detection Result based on Inception V3')
plt.show()

# DR Classification

if prediction == 1:
    print('Eye condition is normal')
    print('Quit')

elif prediction == 0:

    IMG_SIZE = 224
    IM = np.array(FI)
    img_array = cv2.cvtColor(IM, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    N = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    OP2 = load_model(
        r'C:\Users\Computer\Desktop\project-docs\DR app\Model2.h5')
    prediction = OP2.predict(N)

    class_labels = ['Exudates', 'Hemorrhages', 'Microaneurysms']
    prediction = np.argmax(prediction, axis=-1)
    print(prediction)

    print(class_labels[prediction[0]])
    label = class_labels[prediction[0]]

    font = cv2.FONT_HERSHEY_SIMPLEX
    orgin = (550, 300)
    fontScale = 10
    color = (0, 255, 0)
    thickness = 10
    image = cv2.putText(IM, label, orgin, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    # fig5 = plt.figure()
    plt.grid(False)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('DR Classification Result based on Xception')
    plt.show()

print('completed')

exit()
