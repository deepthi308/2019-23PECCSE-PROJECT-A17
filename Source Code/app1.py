# Web application for DR detection and classification

from mlxtend.plotting import plot_confusion_matrix
import pandas as pd  # to read and manipulating data
from sklearn import metrics
import seaborn as sns  # for better visualization of graph with the help of Matplotlib
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import base64
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


st.markdown('<h1 style="color:black;">Diabetic Retinopathy Detection and Classification</h1>',
            unsafe_allow_html=True)

upload = st.file_uploader('Insert image for classification', type=[
                          'png', 'jpg', 'jpeg'])
c1, c2, c3 = st.columns(3)
prediction1 = []

# Upload Image

if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c1.header('Input Image')
    c1.image(im)

    FI = cv2.medianBlur(img, 3)

if st.button('Submit'):

    # DR detection

    OP1 = load_model(
        r'C:\Users\Computer\Desktop\project-docs\DR app\Model1.h5')

    IMG_SIZE = 224
    IM = np.array(FI)
    img_array = cv2.cvtColor(IM, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    N = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    prediction = OP1.predict(N)

    class_labels = ['Diabetic Retinopathy', 'Normal']
    prediction1 = np.argmax(prediction, axis=-1)

    label = class_labels[prediction1[0]]
    c2.header('Detection Result')
    # c2.subheader('DR Detection :')
    c2.write(label)

# DR classification

if prediction1 == 1:
    c3.header('Classification Result')
    c3.subheader('DR Type :')
    c3.write('Eye is in normal condition')
    c3.write('Exit')

elif prediction1 == 0:

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

    label = class_labels[prediction[0]]
    c3.header('Classification Result')
    # c3.subheader('DR Type :')
    c3.write('Eye is in abnormal condition')
    c3.write(label)

else:
    st.write()

# Performance Measure for DR Detection


st.header('Performance Analysis for DR Detection')

test_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\FDATASET\PA'
testdata = ImageDataGenerator()
test_data_gen = testdata.flow_from_directory(
    directory=test_dir, target_size=(224, 224), shuffle=False, class_mode='binary')
batch_size = 32
OP = load_model(r'C:\Users\Computer\Desktop\project-docs\DR app\Model1.h5')
Y_pred = OP.predict(test_data_gen, test_data_gen.samples / batch_size)
test_preds = np.argmax(Y_pred, axis=1)
test_trues = test_data_gen.classes

confusion_matrix = metrics.confusion_matrix(test_preds, test_trues)
lang = ['Diabetic Retinopathy', 'Normal']
conf_matrix_df = pd.DataFrame(confusion_matrix, columns=lang, index=lang)
plot_confusion_matrix(conf_mat=confusion_matrix,
                      class_names=lang,
                      show_absolute=False,
                      show_normed=True,
                      colorbar=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.text('Model Report:\n    ' + classification_report(test_trues, test_preds))


# Performance Measure for DR Classification

st.header('Performance Analysis for DR Classification')

test_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\SDATASET\PA'
testdata = ImageDataGenerator()
test_data_gen = testdata.flow_from_directory(directory=test_dir, target_size=(
    224, 224), shuffle=False, class_mode='categorical')
batch_size = 32
OP1 = load_model(r'C:\Users\Computer\Desktop\project-docs\DR app\Model2.h5')
Y_pred = OP1.predict(test_data_gen, test_data_gen.samples / batch_size)
test_preds = np.argmax(Y_pred, axis=1)
test_trues = test_data_gen.classes

confusion_matrix = metrics.confusion_matrix(test_preds, test_trues)
lang = ['Exudates', 'Hemorrhages', 'Microaneurysms']
conf_matrix_df = pd.DataFrame(confusion_matrix, columns=lang, index=lang)
plot_confusion_matrix(conf_mat=confusion_matrix,
                      class_names=lang,
                      show_absolute=False,
                      show_normed=True,
                      colorbar=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.text('Model Report:\n    ' + classification_report(test_trues, test_preds))
