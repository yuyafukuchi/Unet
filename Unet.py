
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as trans
import tensorflow as tf
import keras.backend as K
from PIL import Image, ImageOps,ImageEnhance
from keras.models import Model
from keras.layers import Input,MaxPooling2D,UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization, Activation, Dropout
from keras.optimizers import Adam,Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class UNet(object):
    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.output_channel_count = 1
        self.firstlayer_filter_count = 64
        self.kernel_initializer = "he_normal"
        self.kernel_size = (3,3)
        self.pool_size = (2,2)
        self.concatenate_axis = 3
        self._create_model()

    def _create_model(self):
        inputs = Input(shape=self.input_shape)

        conv1 = Conv2D(self.firstlayer_filter_count, kernel_size=self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(inputs)
        conv1 = Conv2D(self.firstlayer_filter_count, kernel_size=self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=self.pool_size)(conv1)

        conv2 = Conv2D(self.firstlayer_filter_count*2, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(pool1)
        conv2 = Conv2D(self.firstlayer_filter_count*2, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=self.pool_size)(conv2)
        
        conv3 = Conv2D(self.firstlayer_filter_count*4, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(pool2)
        conv3 = Conv2D(self.firstlayer_filter_count*4, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=self.pool_size)(conv3)

        conv4 = Conv2D(self.firstlayer_filter_count*8, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(pool3)
        conv4 = Conv2D(self.firstlayer_filter_count*8, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=self.pool_size)(conv4)

        conv5 = Conv2D(self.firstlayer_filter_count*16, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(pool4)
        conv5 = Conv2D(self.firstlayer_filter_count*16, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv5)
        conv5 = BatchNormalization()(conv5)
        pool5 = MaxPooling2D(pool_size=self.pool_size)(conv5)

        bridge = Conv2D(self.firstlayer_filter_count*32, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(pool5)
        bridge = Conv2D(self.firstlayer_filter_count*32, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(bridge)
        bridge = BatchNormalization()(bridge)

        up6 = UpSampling2D(size = self.pool_size)(bridge)
        up6 =  Conv2D(self.firstlayer_filter_count*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up6)
        merge6 = concatenate([conv5,up6], axis = self.concatenate_axis)
        conv6 = Conv2D(self.firstlayer_filter_count*16, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(merge6)
        conv6 = Conv2D(self.firstlayer_filter_count*16, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = UpSampling2D(size = self.pool_size)(conv6)
        up7 =  Conv2D(self.firstlayer_filter_count*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7)
        merge7 = concatenate([conv4,up7], axis = self.concatenate_axis)
        conv7 = Conv2D(self.firstlayer_filter_count*8, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(merge7)
        conv7 = Conv2D(self.firstlayer_filter_count*8, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = UpSampling2D(size = self.pool_size)(conv7)
        up8 =  Conv2D(self.firstlayer_filter_count*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up8)
        merge8 = concatenate([conv3,up8], axis = self.concatenate_axis)
        conv8 = Conv2D(self.firstlayer_filter_count*4, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(merge8)
        conv8 = Conv2D(self.firstlayer_filter_count*4, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = UpSampling2D(size = self.pool_size)(conv8)
        up9 =  Conv2D(self.firstlayer_filter_count*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up9)
        merge9 = concatenate([conv2,up9], axis = self.concatenate_axis)
        conv9 = Conv2D(self.firstlayer_filter_count*2, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(merge9)
        conv9 = Conv2D(self.firstlayer_filter_count*2, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv9)
        conv9 = BatchNormalization()(conv9)

        up10 = UpSampling2D(size = self.pool_size)(conv9)
        up10 =  Conv2D(self.firstlayer_filter_count, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up10)
        merge10 = concatenate([conv1,up10], axis = self.concatenate_axis)
        conv10 = Conv2D(self.firstlayer_filter_count, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(merge10)
        conv10 = Conv2D(self.firstlayer_filter_count, kernel_size = self.kernel_size, activation = 'relu', padding = 'same', kernel_initializer = self.kernel_initializer)(conv10)
        conv10 = BatchNormalization()(conv10)

        conv11 = Conv2D(1, kernel_size = (1,1), activation = 'sigmoid')(conv10)

        self.UNET = Model(input=inputs, output=conv11)

    def get_model(self):
        return self.UNET

Path = "./"
TestPath = "./TestImage" 
InputPath = "./Train"
LabelPath = "./TrainLabel"
LabelAnswerPath = "./TestLabel"
SavePath  = "./unet_weights.hdf5" 

Threshold = 0.5
IMAGE_SIZE = 256
CHANNEL_COUNT = 3
INPUT_SHAPE = (IMAGE_SIZE,IMAGE_SIZE,CHANNEL_COUNT)
BATCH_SIZE = 4
NUM_EPOCH = 64

MOLPHOLOGY_KERNELSIZE = 8
MINIMUM_AREA_SIZE = 100 

# 値を0から1に正規化する関数
def easy_normalize(image):
    image = image/255
    return image

# 標準化(平均を0、分散を1に)
def zscore_standardization(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    return (x-xmean)/xstd

# 正規化(最大値が1,最小値が0)
def min_max_normalization(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x-x_min)/(x_max-x_min)

# 値を0から255に戻す関数
def denormalize_y(image):
    image = image*255
    return image

# 元画像をRGBで読み込む関数
def load_X(folder_path):
    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, CHANNEL_COUNT), np.float32)

    for i, image_file in enumerate(image_files):
        image = Image.open(folder_path + os.sep + image_file).resize((IMAGE_SIZE,IMAGE_SIZE)).convert("RGB")
        image = np.asarray(image)

        images[i] = min_max_normalization(image)
    return images, image_files

# ラベル画像を読み込む関数
def load_Y(folder_path):
    #MEMO: labelmeでアノテーションしたデータはPILじゃないと読み込めない
    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files),  IMAGE_SIZE,IMAGE_SIZE,1), np.uint8)

    for i, image_file in enumerate(image_files):
        image = Image.open(folder_path + os.sep + image_file).resize((IMAGE_SIZE,IMAGE_SIZE)).convert("L")
        image = np.asarray(image)
        image = (image!=0)
        images[i] = image[:, : ,np.newaxis]
    return images

# IoUを計算する関数
def iou(y_true,y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    
    intersection = K.sum(y_true * y_pred)
    return intersection / (K.sum(y_true) + K.sum(y_pred) + 1 - intersection)

# 画像水増し関数
# nums: (何倍にするか)-1
def augmentation(X_train,Y_train,nums):
    if len(X_train) != len(Y_train):
      raise Exception  

    data_gen_args = dict(
                        horizontal_flip=True,
                        vertical_flip=True,
                        width_shift_range=0.1,
                        height_shift_range=0.1)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_generator = image_datagen.flow(
        X_train,
        batch_size = 1,
        seed=seed)
    mask_generator = mask_datagen.flow(
        Y_train,
        batch_size = 1,
        seed=seed)
    
    new_X = []
    new_Y = []
    for _ in range(len(X_train)):
      for _ in range(nums):
        new_X.append(next(image_generator)[0])
        new_Y.append(next(mask_generator)[0])
    new_X = np.concatenate([X_train,np.asarray(new_X)])
    new_Y = np.concatenate([Y_train,np.asarray(new_Y)])
    return new_X,new_Y

# U-Netのトレーニングを実行する関数
def train_unet():
    X_train, _ = load_X(TrainPath)
    Y_train = load_Y(TrainLabelPath)    
    X_train,Y_train = augmentation(X_train,Y_train,9)

    X_test,_ = load_X(TestPath)
    Y_test = load_Y(TestLabelPath)

    # U-Netの生成
    network = UNet(input_shape=INPUT_SHAPE)

    model = network.get_model()
    model.summary()
    model.compile(
        loss="binary_crossentropy", 
        optimizer=Adam(), 
        metrics=[iou]
    )

    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH,validation_data=(X_test, Y_test))

    model.save_weights(SavePath)
    return history

# trainを実行し可視化する関数
def visualize_train():
    history = train_unet()

    # 訓練経過の可視化
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.ylabel('iou')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict():
    Threshold = 0.5
    X_test, file_names = load_X(TestPath)

    network = UNet(input_shape=INPUT_SHAPE)
    model = network.get_model()
    model.load_weights(SavePath)

    Y_pred = model.predict(X_test, BATCH_SIZE)
    Y_pred = (Y_pred>=Threshold)
    Y_pred = Y_pred.astype(np.uint8)

    # # 保存
    PredictImageSaveFolder = "predict1/"
    for i, y in enumerate(Y_pred):
        img = cv2.imread(TestPath + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0])) 
        basename_without_ext = os.path.splitext(file_names[i])[0]
        cv2.imwrite(Path + PredictImageSaveFolder + 'prediction_' + basename_without_ext + '.png', denormalize_y(y))
    
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax.imshow(np.squeeze(denormalize_y(Y_pred[1])))
    # ax2.imshow(X_test[1])

def BFS(img):
    """
    予想値の二次元配列から対象領域をBFSで探し,それぞれの領域の重心位置を返す

    Parameters
    ----------
    img : List[List[int]]
        重心検出したい予想値の二次元配列

    Returns
    -------
    G : List[(int,int)]
        各領域の重心座標たち
    """
    # 幅優先探索
    H = img.shape[0] 
    W = img.shape[1]
    isVisited = [[False]*W for _ in range(H)]
    G = []
    li_opening = img.tolist()

    from collections import deque
    for i in range(H):
      for j in range(W):
        if li_opening[i][j] == 0 or isVisited[i][j]:
          continue
        target_rectangle = []
        queue = deque([(i,j)])

        while queue:
          y,x = queue.popleft()
          if 0<=y<H and 0<=x<W:
            if isVisited[y][x]==False and li_opening[y][x]!=0:
              target_rectangle.append((y,x))
              isVisited[y][x] = True

              for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
                  queue.append((y+dy,x+dx))
        
        LEN = len(target_rectangle) 
        if LEN <= MINIMUM_AREA_SIZE:
          continue

        sum_i = 0
        sum_j = 0
        for y,x in target_rectangle:
          sum_i += y
          sum_j += x
        G.append((sum_i//LEN,sum_j//LEN))
        
    return G

def find_center_of_gravity(img_path: str):
    """
    ダイカストの正方形領域の重心位置を返す.またその重心位置をプロット

    Parameters
    ----------
    img_path : str
        予想したいダイカスト画像の保存先のパス
    
    Returns
    -------
    G : List[(int,int)]
        各領域の重心座標たち
    """
    ori_image = Image.open(img_path)
    image = ori_image.resize((IMAGE_SIZE,IMAGE_SIZE))
    ori_image = np.asarray(ori_image)

    rgb_image = np.asarray(image.convert("RGB"))
    image = min_max_normalization(rgb_image)
    image = image[np.newaxis,:,:,:]

    network = UNet(input_shape=INPUT_SHAPE)
    model = network.get_model()
    model.load_weights(SavePath) 

    pred = model.predict(image, BATCH_SIZE)
    pred = (pred>=Threshold).astype(np.uint8)[0]
    pred = cv2.resize(pred, (ori_image.shape[1], ori_image.shape[0])) 

    # kernel = np.ones((MOLPHOLOGY_KERNELSIZE,MOLPHOLOGY_KERNELSIZE),np.uint8)

    # モルフォロジー変換のオープニング処理を施す
    # 参考: http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    # opening = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    G = BFS(pred)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    ax.imshow(ori_image)
    for y,x in G:
      ax.plot(x,y,color='r',marker='.',markersize=20)
    plt.show()
    return G

if __name__ == "__main__":
    # print(find_center_of_gravity(Path+'TestImage/img43.png'))
    predict()