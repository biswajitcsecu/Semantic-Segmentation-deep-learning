#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Common
import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf

# Data 
import tensorflow.data as tfd
import tensorflow.image as tfi
from tensorflow.keras.utils import load_img, img_to_array

# Data Visualization
import matplotlib.pyplot as plt

# Model
from keras.layers import ReLU
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import AveragePooling2D
from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from tensorflow.keras.applications import ResNet50

# Callbacks 
from keras.callbacks import Callback, ModelCheckpoint

# Model Visualization
from tensorflow.keras.utils import plot_model

# Model Insights 
from tf_explain.core.grad_cam import GradCAM


# In[23]:


train_path = 'CamVid/train/'
valid_path = 'CamVid/val/'
test_path = 'CamVid/test/'


# In[24]:


def load_image(path, IMAGE_SIZE=256):
    image = load_img(path)
    image = img_to_array(image)
    image = tfi.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image = image/255.
    return image

def load_images(path_list):
    images = np.zeros(shape=(len(path_list), 256, 256, 3))
    for idx, path in tqdm(enumerate(path_list), desc="Loading"):
        image = load_image(path)
        images[idx] = image
    return images
        
def load_data(root_path, data, trim=None):
    
    # Generate The Image paths
    image_paths = glob(root_path  + '*.png')
    
    label_map_paths = []
    for path in image_paths:
        label_path = path.replace(f"{data}/", f"{data}_labels/")
        label_path = label_path.replace(".png","_L.png")
        label_map_paths.append(label_path)
    
    if trim is None:
        trim = len(image_paths)
    
    
    # Load the Images
    image_paths = image_paths[:trim]
    label_map_paths = label_map_paths[:trim]
    images = load_images(image_paths)
    label_maps = load_images(label_map_paths)
    
    return images, label_maps


# In[25]:


# Params
AUTO = tfd.AUTOTUNE
BATCH_SIZE = 8


# In[26]:


# Load Data
train_images, train_label_maps = load_data(train_path, data='train')

# Converting to TF Data
train_ds = tfd.Dataset.from_tensor_slices((train_images, train_label_maps))
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(AUTO)


# In[27]:


# Load Data
valid_images, valid_label_maps = load_data(valid_path, data='val')

# Converting to TF Data
valid_ds = tfd.Dataset.from_tensor_slices((valid_images, valid_label_maps))
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=True)
valid_ds = valid_ds.prefetch(AUTO)


# In[28]:


# Load Data
test_images, test_label_maps = load_data(test_path, data='test', trim=50)

# Converting to TF Data
test_ds = tfd.Dataset.from_tensor_slices((test_images, test_label_maps))
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=True)
test_ds = test_ds.prefetch(AUTO)
len(test_ds)


# In[29]:


def show_maps(data, n_images=10, model=None, SIZE=(20,10), ALPHA=0.5, explain=False):
    
    # plot Configurations
    if model is not None:
        n_cols = 4
    else:
        n_cols = 3
    
    # Select the Data
    images, label_maps = next(iter(data))
    
    if model is None:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')

            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 3)
            plt.imshow(image)
            plt.imshow(label_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()
    elif explain:
        n_cols = 5
        exp = GradCAM()
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Select Image and Label Map
            images, label_maps = valid_images, valid_label_maps
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]
            pred_map = model.predict(image[np.newaxis, ...])[0]
            
            # Grad Cam
            cam = exp.explain(
                validation_data=(image[np.newaxis,...], label_map),
                class_index=1,
                layer_name='TopCB-2',
                model=model
            )
            
            # Figure Size
            plt.figure(figsize=SIZE)

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Original Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')
            
            # Plot Predicted Label Map
            plt.subplot(1, n_cols, 3)
            plt.imshow(pred_map)
            plt.title('Predicted Label Map')
            plt.axis('off')
            
            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 4)
            plt.imshow(image)
            plt.imshow(pred_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')
            
            # Plot Grad Cam
            plt.subplot(1, n_cols, 5)
            plt.imshow(cam)
            plt.title("Grad Cam")
            plt.axis('off')

            # Final Show
            plt.show()
        
    else:
        # Create N plots where N = Number of Images
        for image_no in range(n_images):

            # Figure Size
            plt.figure(figsize=SIZE)

            # Select Image and Label Map 
            id = np.random.randint(len(images))
            image, label_map = images[id], label_maps[id]
            pred_map = model.predict(image[np.newaxis, ...])[0]

            # Plot Image 
            plt.subplot(1, n_cols, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            # Plot Original Label Map
            plt.subplot(1, n_cols, 2)
            plt.imshow(label_map)
            plt.title('Original Label Map')
            plt.axis('off')
            
            # Plot Predicted Label Map
            plt.subplot(1, n_cols, 3)
            plt.imshow(pred_map)
            plt.title('Predicted Label Map')
            plt.axis('off')
            
            # Plot Mixed Overlap
            plt.subplot(1, n_cols, 4)
            plt.imshow(image)
            plt.imshow(pred_map, alpha=ALPHA)
            plt.title("Overlap")
            plt.axis('off')

            # Final Show
            plt.show()


# In[30]:


show_maps(data=train_ds)


# In[31]:


class ConvBlock(Layer):
    
    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer='he_normal', dilation_rate=dilation_rate, use_bias=True),
            BatchNormalization(),
            ReLU()
        ])
        
    def call(self, X): return self.net(X)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "filters":self.filters, "kernel_size":self.kernel_size, "dilation_rate":self.dilation_rate}


# In[32]:


def AtrousSpatialPyramidPooling(X):
    _, height, width, _ = X.shape
    
    y = AveragePooling2D(pool_size=(height, width), name="ASPP-AvgPool")(X)
    y = ConvBlock(kernel_size=1, name="ASPP-ImagePool")(y)
    image_pool = UpSampling2D(size=(height//y.shape[1], width//y.shape[2]), interpolation='bilinear', name="ASPP-UpSample")(y)
        
    conv_1 = ConvBlock(kernel_size=1, dilation_rate=1, name="ASPP-Conv1")(X)
    conv_6 = ConvBlock(kernel_size=3, dilation_rate=6, name="ASPP-Conv6")(X)
    conv_12 = ConvBlock(kernel_size=3, dilation_rate=12, name="ASPP-Conv12")(X)
    conv_18 = ConvBlock(kernel_size=3, dilation_rate=18, name="ASPP-Conv18")(X)
    
    concat = Concatenate(axis=-1, name="ASPP-Concat")([image_pool, conv_1, conv_6, conv_12, conv_18])
    out = ConvBlock(kernel_size=1, name="ASPP-Output")(concat)
    
    return out



# In[33]:


# Input Layer
IMAGE_SIZE = 256
InputL = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="Input-Layer")

# Base Model
resnet50 = ResNet50(include_top=False, input_tensor=InputL, weights='imagenet')

# DCNN Output
DCNN = resnet50.get_layer('conv4_block6_2_relu').output
ASPP = AtrousSpatialPyramidPooling(DCNN)
ASPP = UpSampling2D(size=(IMAGE_SIZE//4//ASPP.shape[1], IMAGE_SIZE//4//ASPP.shape[2]), interpolation='bilinear', name="Atrous-Upscale")(ASPP)

# Low Level Features
LLF = resnet50.get_layer('conv2_block3_2_relu').output
LLF = ConvBlock(filters=48, kernel_size=1, name="LLF-ConvBlock")(LLF)

# Combine
combine = Concatenate(axis=-1, name="Combine-Features")([ASPP, LLF])
y = ConvBlock(name="TopCB-1")(combine)
y = ConvBlock(name="TopCB-2")(y)
y = UpSampling2D(size=(IMAGE_SIZE//y.shape[1], IMAGE_SIZE//y.shape[1]), interpolation='bilinear', name="Top-UpSample")(y)
LabelMap = Conv2D(filters=3, kernel_size=1, strides=1, activation='sigmoid', padding='same', name="OutputLayer")(y)

# model 
model = Model(InputL, LabelMap, name="DeepLabV3-Plus")
model.summary()


# In[34]:


plot_model(model, "DeepLabV3+.png", dpi=96, show_shapes=True)


# In[36]:


#Training

class ShowProgress(Callback):
    def on_epoch_end(self, epochs, logs=None):
        show_maps(data=valid_ds, model=self.model, n_images=1)


# In[37]:


model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)


# In[ ]:


# Callbacks 
cbs = [ModelCheckpoint("DeepLabV3+.h5",save_best_only=False),ShowProgress()]
nepochs=50
 
history = model.fit(train_ds, validation_data=valid_ds, epochs=nepochs, callbacks=cbs)



# In[ ]:




