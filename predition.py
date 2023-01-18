#from Mascara import Mascara
# 
#import keras,os
#from keras.models import Sequential
#from keras.layers import Dense,  Conv2D, MaxPool2D , Flatten
#from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import normalize
import tensorflow as tf
#from tensorflow.keras.optimizers import Adam

#from keras import backend as K
#from keras.layers import Layer,InputSpec
#import keras.layers as kl
from keras.models import load_model

import numpy as np
#import glob, os
#import re
#from PIL import Image
import matplotlib.pyplot as plt
import cv2


class_names = ['fertil', 'infertil']
#i=0

def resize(imlist):
    maxsize = 180,180
    print(len(imlist))
    print("###########################2")
    #cv2.imshow("TEST",imlist[0])
    data  = []
    for fl in imlist:
        try:
            # print(type(fl))
            # print(fl.shape)
            image = cv2.resize(fl, (maxsize))
        # print(len(image))

            #image=tf.keras.preprocessing.image.load_img(file, color_mode='rgb', target_size= (maxsize))
            image=np.array(image)
           # print(image.shape)
            data.append(image)
        except Exception:
            pass

    data = np.array(data)
    print("###########################3")
    print(len(data))
    return data


def pred(imlist):
   # print("###########################1")
    img_test = resize(imlist)

    dir_model = r"C:\Users\Igor Santos\Documents\data\App\Model\Densenet_aug128.h5"
   # print(dir_model)
    loaded_model = load_model(dir_model)

    """#### Compile"""
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    print(img_test.shape)
    """Predict"""
    y_pred =  loaded_model.predict(img_test)
    #y_pred = np.round(abs(pred)).astype('uint8')
    #y_true = label_test

    y_pred2=[]
    for i in range(len(img_test)):
        y_pred2.append(np.around(y_pred[i][1]).astype('uint8'))

    #print(y_pred2)
    cont_fer = 0
    cont_inf = 0
    #pred_infert=pred_fertil=[]
    for i in range(len(img_test)):
        if y_pred2[i]==0:
            cont_fer += 1 
           # pred_fertil
        else:
            cont_inf+= 1
            #pred_infert
    #print(fertil,infertil)   
    #plot(y_pred2,img_test)
    return (y_pred2,cont_fer,cont_inf)

#PLOT
def plot(pred,patch):
    import os
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    start,end = 0,30
    predict =pred[start:end]
    img = patch[start:end]

    print('OKAY')

    plt.figure(figsize=(16,14))
    for i in range(30):
        print(i)
        plt.subplot(6, 5, i+1)
        plt.axis('off')
        # plt.imshow(img[i])
        # plt.title(class_names[pred[i]])

        plt.imshow(img[i])
        plt.title(class_names[predict[i]])
        plt.show()