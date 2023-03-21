# Web application for DR detection and classification

import streamlit as st
import base64
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Background Image

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-size: cover;
    background-position: center;
   
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('./images/33.jpeg')


title = '<p style="font-family:Brush Script MT; color:white; font-size: 42px;">Diabetic Retinopathy Detection and Classification</p>'
st.markdown(title, unsafe_allow_html=True)

# st.markdown('<h1 style="color:black;">Diabetic Retinopathy Detection and Classification</h1>', unsafe_allow_html=True)

upload = st.file_uploader('Insert image for classification', type=[
                          'png', 'jpg', 'jpeg'])
# c1, c2, c3 = st.columns(3)
prediction1 = []

# Upload Image

if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.markdown('<h3 style="color:white;">Input Image</h3>',
                unsafe_allow_html=True)
    #  st.subheader('<h1>Input Image</h1>')
    st.image(im)

    FI = cv2.medianBlur(img, 3)

if st.button('Submit'):

    # DR detection

    OP1 = load_model(
        r'./Model1.h5')

    IMG_SIZE = 224
    IM = np.array(FI)
    img_array = cv2.cvtColor(IM, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    N = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    prediction = OP1.predict(N)

    class_labels = ['Diabetic Retinopathy', 'Normal']
    prediction1 = np.argmax(prediction, axis=-1)

    label = class_labels[prediction1[0]]
    # st.subheader('Detection Result :')
    st.markdown('<h3 style="color:white;">Detection Result :</h3>',
                unsafe_allow_html=True)
    # c2.subheader('DR Detection :')
    st.write(label)

# DR classification

if prediction1 == 1:
    st.markdown('<h3 style="color:white;">Detection Result :</h3>',
                unsafe_allow_html=True)
    # st.subheader('Classification Result : ')
    # c3.subheader('DR Type :')
    # c3.write('Eye is in normal condition')
    st.write('None')

elif prediction1 == 0:

    IMG_SIZE = 224
    IM = np.array(FI)
    img_array = cv2.cvtColor(IM, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    N = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    OP2 = load_model(
        r'./Model2.h5')

    prediction = OP2.predict(N)

    class_labels = ['Exudates', 'Hemorrhages', 'Microaneurysms']
    prediction = np.argmax(prediction, axis=-1)

    label = class_labels[prediction[0]]
    st.markdown('<h3 style="color:white;">Classification Result :</h3>',
                unsafe_allow_html=True)
    # c3.subheader('DR Type :')
    # c3.write('Eye is in abnormal condition')
    st.write(label)

else:
    st.write()
