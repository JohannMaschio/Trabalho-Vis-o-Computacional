# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:06:19 2022

@author: Johann
"""
# Importando os modelos necessarios #####
from sklearn.preprocessing import StandardScaler
import os
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
from PIL import Image
import numpy as np
from sklearn import metrics

# Carregando as imagens #####
IMG_WIDTH = 200
IMG_HEIGHT = 200
img_folder = r'C:/Base_trabalho/'
def create_dataset(img_folder):    
    img_data_array = {}
    img_data_array['data'] = []
    img_data_array['classe'] = []
    for dir in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir)):       
            image_path = os.path.join(img_folder, dir,  file)
            image = np.array(Image.open(image_path))
            image = resize(image, (IMG_HEIGHT,IMG_WIDTH))
            img_data_array['data'].append(image)
            img_data_array['classe'].append(dir)
    return img_data_array 
img_data = create_dataset(img_folder)

# Imprimindo a imagem #####
#plt.imshow(img_data['data'][1])

# Dividindo o dataset em treino e teste #####
seed = random.seed(9)      # just so you get the same answers as me #####

x = np.array(img_data['data'])
y = np.array(img_data['classe'])

img_train, img_test, y_train, y_test = train_test_split(np.array(x), 
                                                        np.array(y), 
                                                        test_size=0.20, 
                                                        shuffle=True, 
                                                        random_state=9)
   
class Hog_trans(BaseEstimator, TransformerMixin):
    def __init__(self, y=None, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2)):    
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def local_hog(self, X):
        return hog(
            X,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,)

    def transform(self, X, y=None):
        try:
            return np.array([self.local_hog(img) for img in X])
        except:
            return np.array([self.local_hog(img) for img in X])
   
hog_t = Hog_trans(pixels_per_cell=(8, 8), cells_per_block=(2,2), orientations=9,)

# Process train data #####
train_hog = hog_t.fit_transform(img_train)

# Process test data #####
test_hog = hog_t.transform(img_test)  
    
#### Fit and Prediction ####
classifier = RandomForestClassifier(max_depth=4)
classifier.fit(train_hog, y_train)
y_predicted = classifier.predict(test_hog)

# Print the classification report #####
names = ['Neg', 'Pos']
print(metrics.classification_report(y_test, y_predicted, target_names=names))

# Print and plot the confusion matrix #####
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)


# Calculando as metreicas #####
tn, fp, fn, tp = cm.ravel()   
accuracy = (tp+tn) / (tp+tn+fp+fn)
precision = tp / (tp+fp)
sensitivity = tp / (tp+fn)
specificity = tn / (tn+fp)
f1 = tp / (tp + ((fp+fn)*(1/2)))

# Imprimindo as metricas #####
print("F1 Score: ", f1)
print("Precision Score: ", precision)
print("Accuracy: ", accuracy)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
