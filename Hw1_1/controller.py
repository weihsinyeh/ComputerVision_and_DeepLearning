from pickle import TRUE
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2, os 
import numpy as np
import matplotlib.pyplot as plt
from UI import Ui_Dialog
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def nothing(x):
    pass
class MainWindow_controller(QtWidgets.QMainWindow):
    filepath1 = ""
    filepath2 = ""
    alpha = 0
    beta = 0
    
    def __init__(self, parent=None):
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        title = "2022 Opencvdl Hw1"
        self.setWindowTitle(title)
        self.setup_control()

    def setup_control(self):
        self.ui.ColorSeparation.setEnabled(False)
        self.ui.ColorTransform.setEnabled(False)
        self.ui.ColorDetection.setEnabled(False)
        self.ui.Blending.setEnabled(False)
        self.ui.GaussianBlur.setEnabled(False)
        self.ui.BilateralFilter.setEnabled(False)
        self.ui.MedianFilter.setEnabled(False)
        self.ui.LoadImage2.setEnabled(False)

        self.ui.LoadImage1.clicked.connect(self.open_file1) 
        self.ui.LoadImage2.clicked.connect(self.open_file2)
        self.ui.ColorSeparation.clicked.connect(self.color_separation)
        self.ui.ColorTransform.clicked.connect(self.color_transform)
        self.ui.ColorDetection.clicked.connect(self.color_detection)
        self.ui.Blending.clicked.connect(self.blending)
        self.ui.GaussianBlur.clicked.connect(self.GaussianBlur)
        self.ui.BilateralFilter.clicked.connect(self.BilateralBlur)
        self.ui.MedianFilter.clicked.connect(self.MedianFilter)

    # 2.1 Gaussian Blur
    def GaussianBlur(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("gaussian_blur",img)
        min = 0
        max = 10
        def change(value): 
            output = cv2.GaussianBlur(img, (2*value+1, 2*value+1), 0)
            cv2.setTrackbarPos('magnitude','gaussian_blur',value)
            cv2.imshow("gaussian_blur",output)
        cv2.createTrackbar('magnitude','gaussian_blur',min,max,change)
        cv2.waitKey(0)

    # 2.2 Bilateral Filter
    def BilateralBlur(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("bilateral_filter",img)
        min = 0
        max = 10
        def change(value): 
            output = cv2.bilateralFilter(img, 2*value+1,90,90)
            cv2.setTrackbarPos('magnitude','bilateral_filter',value)
            cv2.imshow("bilateral_filter",output)

        cv2.createTrackbar('magnitude','bilateral_filter',min,max,change)
        cv2.imshow("bilateral_filter",img)
        cv2.waitKey(0)
      
        

    # 2.3 Median Filter
    def MedianFilter(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("median_filter",img)
        min = 0
        max = 10
        def change(value): 
            output = cv2.medianBlur(img, 2*value+1)
            cv2.setTrackbarPos('magnitude','median_filter',value)
            cv2.imshow("median_filter",output)
           
        cv2.createTrackbar('magnitude','median_filter',min,max,change)
        cv2.imshow("median_filter",img)
        cv2.waitKey(0)
    
    # 1.1 Color Separation
    def color_separation(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("Original",img)
        zero_channel = np.zeros(img.shape[0:2],dtype = "uint8")
        B,G,R = cv2.split(img)
        imgR = cv2.merge([zero_channel,zero_channel,R])
        imgG = cv2.merge([zero_channel,G,zero_channel])
        imgB = cv2.merge([B,zero_channel,zero_channel])
        cv2.imshow("Color Separation Red",imgR)
        cv2.imshow("Color Separation Blue",imgB)
        cv2.imshow("Color Separation Green",imgG)
        cv2.imwrite('./Result/1_1_Color_Separation_Red.jpg', imgR, [cv2.IMWRITE_JPEG_QUALITY, 80])    # 存成 jpg
        cv2.imwrite('./Result/1_1_Color_Separation_Green.jpg', imgG, [cv2.IMWRITE_JPEG_QUALITY, 80])  # 存成 jpg
        cv2.imwrite('./Result/1_1_Color_Separation_Blue.jpg', imgB, [cv2.IMWRITE_JPEG_QUALITY, 80])   # 存成 jpg
        cv2.waitKey(0)

    # 1.2 Color Transformation
    def color_transform(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("Original",img)
        averageWeight = img
        openCVfunction = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow("OpenCV Function",openCVfunction)
        (row, col) = averageWeight.shape[0:2]
        for i in range(row):
            for j in range(col):
                averageWeight[i, j] = sum(averageWeight[i, j]) * 0.33
        cv2.imshow("Average Weighted",averageWeight)
        cv2.imwrite('./Result/1_2_OpenCV_Function.jpg', openCVfunction, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite('./Result/1_2_Average_Weighted.jpg', averageWeight, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # 1.3 Color Detection
    def color_detection(self):
        #img = cv2.imread(filepath1)
        img = cv_imread(filepath1)
        cv2.imshow("Original",img)
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        white_lower_bound = np.array([0,0,200])
        white_upper_bound = np.array([180,20,255])
        green_lower_bound = np.array([40,50,20])
        green_upper_bound = np.array([80,255,255])
        whitemask = cv2.inRange(imgHSV , white_lower_bound , white_upper_bound)
        greenmask = cv2.inRange(imgHSV , green_lower_bound , green_upper_bound)
        white = cv2.bitwise_and(img , img , mask = whitemask)
        green = cv2.bitwise_and(img , img , mask = greenmask)
        cv2.imshow("Color_Detection_White",white)
        cv2.imshow("Color_Detection_Green",green)
        cv2.imwrite('./Result/1_3_Color_Detection_White.jpg', white, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite('./Result/1_3_Color_Detection_Green.jpg', green, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # 1.4 Blending
    def blending(self):
        global alpha,beta
        #img = cv2.imread(filepath1)
        img1 = cv_imread(filepath1)
        #img2 = cv2.imread(filepath2)
        img2 = cv_imread(filepath2)
        h,w, _ = img1.shape
        img2 = cv2.resize(img2,(w,h),interpolation = cv2.INTER_AREA)
        dst = cv2.addWeighted(img1,1,img2,0,10)
        cv2.imshow("Blend",dst)
        cv2.namedWindow('Blend')
        max = 255
        min = 0
        def change(value):
            beta = value / (max - min)
            alpha = ( 1.0 - beta )
            dst = cv2.addWeighted(img1,alpha,img2,beta,10)
            cv2.imshow("Blend",dst)

        cv2.createTrackbar('Blend','Blend',min,max,change)
        cv2.imshow("Blend",dst)
    # Open File 1
    def open_file1(self):
        global filepath1
        filepath1 , ok = QFileDialog.getOpenFileName(self,"Open file","./Image/")  
        if ok:
            #pass
            #else:
            filename1 = os.path.basename(filepath1)
            filename1 = filename1.split('.')[0]
            print(filename1)
            self.ui.Path1.setText(filename1)
            self.ui.ColorSeparation.setEnabled(True)
            self.ui.ColorTransform.setEnabled(True)
            self.ui.ColorDetection.setEnabled(True)
            self.ui.GaussianBlur.setEnabled(True)
            self.ui.BilateralFilter.setEnabled(True)
            self.ui.MedianFilter.setEnabled(True)
            self.ui.LoadImage2.setEnabled(True)
            

    # Open File 2
    def open_file2(self):
        global filepath2
        filepath2,ok = QFileDialog.getOpenFileName(self,"Open file","./Image/")           # start path
        if ok:
            filename2 = os.path.basename(filepath2)
            filename2 = filename2.split('.')[0]
            print(filename2)
            self.ui.Path2.setText(filename2)
            self.ui.Blending.setEnabled(True)
        
