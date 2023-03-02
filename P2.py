# Performance analysis for DR Classification


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import seaborn as sns  # for better visualization of graph with the help of Matplotlib
from sklearn import metrics
import pandas as pd  # to read and manipulating data
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

test_dir = r'C:\Users\Computer\Desktop\project-docs\DR app\SDATASET\PA'
testdata = ImageDataGenerator()
test_data_gen = testdata.flow_from_directory(directory=test_dir, target_size=(
    224, 224), shuffle=False, class_mode='categorical')

batch_size = 32

OP1 = load_model(r'C:\Users\Computer\Desktop\project-docs\DR app\Model2.h5')

Y_pred = OP1.predict(test_data_gen, test_data_gen.samples / batch_size)
test_preds = np.argmax(Y_pred, axis=1)
test_trues = test_data_gen.classes
# Printing class dictionary
print(test_data_gen.class_indices)
print('Classified Result:', test_preds)
print('Ture Result:', test_trues)

confusion_matrix = metrics.confusion_matrix(test_preds, test_trues)
lang = ['Exudates', 'Hemorrhages', 'Microaneurysms']
conf_matrix_df = pd.DataFrame(confusion_matrix, columns=lang, index=lang)
sns.set(font_scale=1.0)
sns.heatmap(conf_matrix_df,
            annot=True)

plt.xlabel('Classified Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.title('Confusion matrix for DR classification')
plt.show()

print(classification_report(test_trues, test_preds))

print('completed')
exit()
