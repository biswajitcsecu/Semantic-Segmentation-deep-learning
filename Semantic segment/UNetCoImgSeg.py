#!/usr/bin/env python
# coding: utf-8

# In[78]:


import os
import sys
import numpy as np
import re
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')
sns.set_style("dark")
sns.set_context("talk")


# In[79]:


# defining u-net

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x



# defining Unet 
def GiveMeUnet(inputImage, numFilters = 16, droupouts = 0.1, doBatchNorm = True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model


# In[80]:


## instanctiating model
inputs = tf.keras.layers.Input((256, 256, 3))
unet = GiveMeUnet(inputs, droupouts= 0.07)
unet.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


# In[81]:


from keras.utils.vis_utils import plot_model
plot_model(unet, to_file='UnetArchitecture.png', show_shapes=True, show_layer_names=True)


# In[82]:


# defining dataLoading 
framObjTrain = {'img' : [], 'mask' : [] }

def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):
    imgNames = os.listdir(imgPath)
    maskNames = []
    
    ## generating mask names
    for mem in imgNames:
        maskNames.append(re.sub('\.png', '_seg0.png', mem))
    
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    for i in range (len(imgNames)):
        try:
            img = plt.imread(imgAddr + imgNames[i]) 
            mask = plt.imread(maskAddr + maskNames[i])
            
        except:
            continue
        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0]) 
        
    return frameObj


# In[60]:


framObjTrain = LoadData( framObjTrain, imgPath = 'Butterfly/images', maskPath = 'Butterfly/masks/' , shape = 256)


# In[61]:


## displaying data loaded 
plt.subplot(1,2,1)
plt.imshow(framObjTrain['img'][1])
plt.subplot(1,2,2)
plt.imshow(framObjTrain['mask'][1],cmap='gray')
plt.show()


# In[83]:


#model fit
nepochs = 10

retVal = unet.fit(np.array(framObjTrain['img']), np.array(framObjTrain['mask']), epochs = nepochs, verbose = 1)


# In[ ]:


#plot performance
plt.plot(retVal.history['loss'], label = 'training_loss')
plt.plot(retVal.history['accuracy'], label = 'training_accuracy')
plt.legend()
plt.grid(True)


# In[ ]:


def predict16 (valMap, model, shape = 256):
    ## getting & proccessing val data
    img = valMap['img']
    mask = valMap['mask']
    mask = mask[0:30]
    
    imgProc = img [0:30]
    imgProc = np.array(img)
    
    predictions = model.predict(imgProc) 

    return predictions, imgProc, mask

def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9,9))
    
    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(1,4,2)
    plt.imshow(predMask,cmap='gray')
    plt.title('Predicted Mask')
    
    plt.subplot(1,4,3)
    plt.imshow(groundTruth,cmap='gray')
    plt.title('actual Mask')
    
    imh = predMask
    imh[imh < 0.5] = 0
    imh[imh > 0.5] = 1
    
    plt.subplot(1,4,4)
    plt.imshow(cv2.merge((imh, imh, imh)) * img)
    plt.title('segmented Image')


# In[ ]:


sixteenPrediction, actuals, masks = predict16(framObjTrain, unet)


# In[ ]:


Plotter(actuals[1], sixteenPrediction[1][:,:,0], masks[1])


# In[ ]:


Plotter(actuals[2], sixteenPrediction[2][:,:,0], masks[2])


# In[ ]:


Plotter(actuals[4], sixteenPrediction[4][:,:,0], masks[4])


# In[ ]:


Plotter(actuals[7], sixteenPrediction[7][:,:,0], masks[7])


# In[ ]:


Plotter(actuals[8], sixteenPrediction[8][:,:,0], masks[8])


# In[ ]:


Plotter(actuals[15], sixteenPrediction[15][:,:,0], masks[15])


# In[ ]:


Plotter(actuals[12], sixteenPrediction[12][:,:,0], masks[12])


# In[ ]:


Plotter(actuals[5], sixteenPrediction[5][:,:,0], masks[5])


# In[ ]:


Plotter(actuals[20], sixteenPrediction[20][:,:,0], masks[20])


# In[ ]:


Plotter(actuals[25], sixteenPrediction[25][:,:,0], masks[25])


# In[ ]:




