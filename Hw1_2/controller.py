from pickle import TRUE
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2, os 
from scipy import signal
from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy  

from UI import Ui_Dialog
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def nothing(x):
    pass
def sobel_1(src,dx, dy, ksize = 5):
        #src = cv2.imread('./Result/3_1_Gaussian_Blur.jpg', cv2.IMREAD_GRAYSCALE)  
        imgSize = src.shape
        #返回结果
        Weighted = np.ones(imgSize)
        Weighted2 = np.ones(imgSize)
        Weighted3 = np.ones(imgSize)
        #dx = 1 则计算Gx  dy = 1 则计算Gy
        #如果都为0，则返回None
        if dx == 0 and dy == 0:
            return None
        #用来计算每个核的6个点
        step = ksize//2
    
        #遍历像素点，这个就不说啦
        for row, row_dat in enumerate(src, step):
            if row >= imgSize[0] - step:
                break
            for col, col_dat in enumerate(row_dat, step):
                if col >= imgSize[1] - step:
                    break
                #计算Gx  为了减少计算量，我们只计算6个点
                if dx == 1:
                    #根据上面的公式计算
                    sobelx = src[row - step][col - step] * -1.0 + src[row - step][col + step] + \
                            src[row ][col - step] * -2.0 + src[row][col + step] * 2.0 + \
                            src[row + step][col - step] * -1.0 + src[row + step][col + step]
                    #保存当前像素点的值
                    Weighted[row][col] = sobelx
            
                if dy == 1:
                    sobely = src[row - step][col - step] * -1.0 + src[row - step][col] * -2.0 + \
                            src[row - step][col + step] * -1.0 + src[row + step][col - step] + \
                            src[row + step][col] * 2.0 + src[row + step][col + step]
                    Weighted2[row][col] = sobely
                if dx == 1 & dy == 1:
                    Weighted3[row][col] = pow(pow(Weighted[row][col],2) + pow(Weighted2[row][col],2),1/2)
        # 同时计算时，返回的结果取绝对值~~~~
        if dx == dy:
            return Weighted3
        #下面的返回结果不需要处理
        if dy == 1:
            return Weighted2
        return Weighted
class MainWindow_controller(QtWidgets.QMainWindow):
    filepath = ""
    
    def __init__(self, parent=None):
        super(MainWindow_controller, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        title = "2022 Opencvdl Hw1"
        self.setWindowTitle(title)
        self.setup_control()

    def setup_control(self):
        self.ui.GaussianBlur.setEnabled(False)
        self.ui.SobelX.setEnabled(False)
        self.ui.SobelY.setEnabled(False)
        self.ui.Magnitude.setEnabled(False)

        self.ui.Resize.setEnabled(False)
        self.ui.Translation.setEnabled(False)
        self.ui.RotationScaling.setEnabled(False)
        self.ui.Shearing.setEnabled(False)
        
        self.ui.LoadImage.clicked.connect(self.open_file) 
       
        self.ui.GaussianBlur.clicked.connect(self.GaussianBlur)
        self.ui.SobelX.clicked.connect(self.SobelX)
        self.ui.SobelY.clicked.connect(self.SobelY)
        self.ui.Magnitude.clicked.connect(self.SobelXY)
        self.ui.Resize.clicked.connect(self.Resize)
        self.ui.Translation.clicked.connect(self.Translation)
        self.ui.RotationScaling.clicked.connect(self.RotationScaling)
        self.ui.Shearing.clicked.connect(self.Shearing)
    # 3.1 Gaussian Blur
    def GaussianBlur(self):
        # convolution
        def convolution(img, kernel):
            (row,col) = img.shape[0:2]
            kernel_size = 3
            new_img_size_row = row - kernel_size + 1
            new_img_size_col = col - kernel_size + 1
            new_img = img
           
            # 經典雙迴圈
            for i in range(new_img_size_row):
                for j in range(new_img_size_col):
                    value = 0
                    # 經典雙雙迴圈 加大不加價(誤)
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            value += img[i+ki][j+kj] * kernel[ki][kj]
                    new_img[i][j] = value
            return new_img
        # convolution    
        global filepath
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2)) #平方相加取絕對值開根號
        
        #Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        img = cv_imread(filepath)

        garyscale = img
        (row, col) = img.shape[0:2]
        r ,g ,b= img[:,:,0],img[:,:,1],img[:,:,2]
        for i in range(row):
            for j in range(col):
                garyscale[i, j] = r[i,j] * 0.2989 + g[i,j] * 0.5870 + b[i,j] * 0.1140
        cv2.imshow("3.1) GrayScale",garyscale)
        cv2.imwrite('./Result/3_1_GrayScale.jpg', garyscale, [cv2.IMWRITE_JPEG_QUALITY, 80])
        new_img = convolution(garyscale,gaussian_kernel)
        new_img=Image.fromarray(new_img)
        img_result = cv2.cvtColor(numpy.asarray(new_img),cv2.COLOR_RGB2BGR) 
     
        cv2.imshow("3.1) Gaussian Blur",img_result)
        cv2.imwrite('./Result/3_1_Gaussian_Blur.jpg', img_result, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.waitKey(0)
    
    # 3.2 SobelX
    def SobelX(self):
        img = cv2.imread('./Result/3_1_Gaussian_Blur.jpg', cv2.IMREAD_GRAYSCALE)  
        img_gx = sobel_1(img,1,0)
        img_gx = cv2.convertScaleAbs(img_gx)
        cv2.imshow("3.2) Sobel X ",img_gx)
        cv2.imwrite('./Result/3_2_Sobel_X.jpg', img_gx, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.waitKey(0)
    # 3.3 SobelY
    def SobelY(self):
        img = cv2.imread('./Result/3_1_Gaussian_Blur.jpg', cv2.IMREAD_GRAYSCALE)
        img_gy = sobel_1(img,0,1)
        img_gy = cv2.convertScaleAbs(img_gy)
        cv2.imshow("3.3) Sobel Y ",img_gy)
        cv2.imwrite('./Result/3_3_Sobel_Y.jpg', img_gy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.waitKey(0)
    # 3.4 SobelXY
    def SobelXY(slef):
        img = cv2.imread('./Result/3_1_Gaussian_Blur.jpg', cv2.IMREAD_GRAYSCALE)
        img_gxy = sobel_1(img,1,1)
        img_gxy = cv2.convertScaleAbs(img_gxy)
        cv2.imshow("3.4) Magnitude",img_gxy)
        cv2.imwrite('./Result/3_4_Magnitude.jpg', img_gxy, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.waitKey(0)

    # 4.1 Color Separation
    def Resize(self):
        img = cv_imread(filepath)
        cv2.imshow("Original",img)
        original_rows,original_cols,original_channels = img.shape
        smalling = img
        smalling = cv2.resize(img, (215, 215))
        rows,cols,channels = smalling.shape
        img[0:rows,0:cols] = smalling

        black = np.zeros((original_rows, original_cols, 3), dtype="uint8")

        black[0:rows,0:cols] = smalling
        cv2.imshow("4.1) Resized Image with black",black)
        cv2.imwrite('./Result/4_1_smalling.jpg', smalling, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.imwrite('./Result/4_1_overlapwithblack.jpg', black, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cv2.waitKey(0)

    # 4.2 Translation
    def Translation(self):
        imgblack = cv_imread('./Result/4_1_overlapwithblack.jpg')
        imgsmall = cv_imread('./Result/4_1_smalling.jpg')
        imgblack[215:430,215:430] = imgsmall
        
        cv2.imshow("4.2) Translate + Overlay",imgblack)
        cv2.imwrite('./Result/4_2_Translation_and_Overlap.jpg', imgblack, [cv2.IMWRITE_JPEG_QUALITY, 80])
        

    # 4.3 RotationScaling
    def RotationScaling(self):
        img = cv_imread('./Result/4_2_Translation_and_Overlap.jpg')

        height = img.shape[0]       # 定義圖片的高度
        width = img.shape[1]        # 定義圖片的寬度
        center = (int(height/2), int(width/2)) # 定義圖片的中心
        angle = 45.0                # 指定旋轉角度
        scale = 0.5                 # 指定縮放比例
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        imageresult = cv2.warpAffine(img, trans, (width, height))
        cv2.imshow("4.3) Rotate and Scale",imageresult)
        cv2.imwrite('./Result/4_3_RotationScaling.jpg', imageresult, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # 4.4 Shearing
    def Shearing(self):
        img = cv_imread('./Result/4_2_Translation_and_Overlap.jpg')
        height = img.shape[0]       # 定義圖片的高度
        width = img.shape[1]        # 定義圖片的寬度
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
        trans = cv2.getRotationMatrix2D(center, 0, 1)
        cv2.imshow("4.4) Shearing",imgresult2)
        cv2.imwrite('./Result/Shearing.jpg', imgresult2, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    # Open File
    def open_file(self):
        global filepath
        filepath , ok = QFileDialog.getOpenFileName(self,"Open file","./Image/")  
        if ok:
           
            filename = os.path.basename(filepath)
            filename = filename.split('.')[0]
            print(filename)
            self.ui.Path.setText(filename)

            self.ui.GaussianBlur.setEnabled(True)
            self.ui.SobelX.setEnabled(True)
            self.ui.SobelY.setEnabled(True)
            self.ui.Magnitude.setEnabled(True)

            self.ui.Resize.setEnabled(True)
            self.ui.Translation.setEnabled(True)
            self.ui.RotationScaling.setEnabled(True)
            self.ui.Shearing.setEnabled(True)
            
        