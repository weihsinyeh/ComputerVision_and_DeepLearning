from pickle import TRUE
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from keras.datasets import cifar10
import cv2, os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy  
from PyQt5 import QtWidgets, QtGui
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import decode_predictions
from UI import Ui_Dialog
from glob import glob
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


x =      ['airplane','automobile','bird',  'cat',  'deer',  'dog','frog','horse','ship','truck']
class MainWindow_controller(QtWidgets.QMainWindow):
    filepath = ""
    def __init__(self, parent=None):
        global model
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        title = "2022 Opencvdl Hw1"
        model  = load_model( 'model2.h5')
        self.setWindowTitle(title)
        self.setup_control()
        self.filepath
    def setup_control(self):
        self.ui.Inference.setEnabled(False)
        self.ui.ShowDataAugumentation.setEnabled(False)
        self.ui.LoadImage.clicked.connect(self.open_file)
        self.ui.ShowTrainImages.clicked.connect(self.ShowTrainImages)
        self.ui.ShowModelStructure.clicked.connect(self.ShowModelStructure)
        self.ui.ShowDataAugumentation.clicked.connect(self.ShowDataAugumentation)
        self.ui.ShowAccuracyandLoss.clicked.connect(self.ShowAccuracyandLoss)
        self.ui.Inference.clicked.connect(self.Inference)
        
    # 5.1 Show Train Images 
    def ShowTrainImages(self):
        filepath = ["./train/0_77.jpg   ", "./train/1_49121.jpg", "./train/2_49049.jpg",
                    "./train/3_1098.jpg ", "./train/4_16195.jpg", "./train/7_29406.jpg",
                    "./train/6_15484.jpg", "./train/9_46656.jpg", "./train/8_7108.jpg "]
        #x =      ['airplane','automobile','bird',  'cat',  'frog',  'horse','frog','truck','ship']
        index=0
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(17, 8))
        for i in range(3):
            for j in range(3):
                img = cv2.imread(filepath[i*3 + j])
                imgresult = img
                img = np.array(img)
                img = img.reshape(1,32,32,3)
                img = img.astype('float32')
                img = img / 255.0
                result = model.predict(img)                
                #axes[i,j].set_title(x[np.argmax(result)] )
                axes[i,j].set_title(x[i*3 + j] )
                axes[i,j].imshow(imgresult)
                axes[i,j].get_xaxis().set_visible(False)
                axes[i,j].get_yaxis().set_visible(False)
                index += 1
        plt.show()

    # 5.2 Show Model Structure
    def ShowModelStructure(self):
        global model
        print(model.summary())
        cv2.waitKey(0)

    # 5.3 Show Data Augumentation
    def ShowDataAugumentation(self):
        img = cv2.imread(self.filepath)
        img = cv2.resize(img, (600, 600)) 
        height = img.shape[0]       # 定義圖片的高度
        width = img.shape[1]        # 定義圖片的寬度

        center = (int(height/2), int(width/2)) # 定義圖片的中心
        angle = 45.0                # 指定旋轉角度
        scale = 0.5                 # 指定縮放比例
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        imageresult = cv2.warpAffine(img, trans, (width, height))
        cv2.imshow("5.3_1) Rotate and Scale1",imageresult)

        center = (int(height/2), int(width/2)) # 定義圖片的中心
        angle = 90.0                # 指定旋轉角度
        scale = 0.5                 # 指定縮放比例
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        imgresult1 = cv2.warpAffine(img,trans,(width,height))
        M_shear = np.array([
            [1,0,0],
            [0.5,1,-(height/4)],
        ],dtype = np.float32)       

        imgresult2 = cv2.warpAffine(imgresult1,M_shear,(width,height))
        cv2.imshow("5.3_2) Shearing",imgresult2)

        # 裁切區域的 x 與 y 座標（左上角）
        x = 100
        y = 100
        w = 250
        h = 150
        imgresult3 = img[y:y+h, x:x+w]
        
        # Shows the image in image viewer
        cv2.imshow("5.3_3) Shearing",imgresult3)
        cv2.waitKey(0)
    # 5.4 Show Accuracy and Loss
    def ShowAccuracyandLoss(slef):
        imgLoss = cv2.imread('./Analysis/Loss_Graph2.png')
        imgAccuracy = cv2.imread('./Analysis/Accuracy_Graph2.png')
        cv2.imshow("5.4) imgLoss",imgLoss)
        cv2.imshow("5.4) imgAccuracy",imgAccuracy)
        cv2.waitKey(0)

    # 5.5 Color Inference
    def Inference(self):
        img = cv2.imread(self.filepath)
        imgresult = cv2.resize(img,(400,400),interpolation = cv2.INTER_CUBIC)
        img = np.array(img)
        img = img.reshape(1,32,32,3)
        img = img.astype('float32')
        img = img / 255.0
        result = model.predict(img)
        
        self.ui.ImageTitle.setText(f"Confidence: {np.max(result)}\n")
        self.ui.ImageTitle_2.setText("Prediction Label: "+x[np.argmax(result)])
        self.mypixmap = QPixmap.fromImage(QImage(imgresult.data,imgresult.shape[1],imgresult.shape[0],QImage.Format_RGB888))
        self.ui.Image.setPixmap(self.mypixmap)
        cv2.waitKey(0)
    
    # Open File
    def open_file(self):
        self.filepath , ok = QFileDialog.getOpenFileName(self,"Open file","./train/")  
        if ok:
            filename = os.path.basename(self.filepath)
            filename = filename.split('.')[0]
            self.ui.Inference.setEnabled(True)
            self.ui.ShowDataAugumentation.setEnabled(True)
            img = cv2.imread(self.filepath)
            imgresult = cv2.resize(img,(400,400),interpolation = cv2.INTER_CUBIC)
            self.mypixmap = QPixmap.fromImage(QImage(imgresult.data,imgresult.shape[1],imgresult.shape[0],QImage.Format_RGB888))
            self.ui.Image.setPixmap(self.mypixmap)
            
        