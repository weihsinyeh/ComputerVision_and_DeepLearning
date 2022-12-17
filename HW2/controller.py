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

class MainWindow_controller(QtWidgets.QMainWindow):
    filepathL = ""
    filepathR = ""
    filefolder = ""
    all_file_name = []
    folder_path =""
    ret = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    imgL = None
    imgR = None
    def __init__(self, parent=None):
        super(MainWindow_controller, self).__init__(parent) 
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        title = "2022 Opencvdl Hw2"
        self.setWindowTitle(title)
        self.setup_control()

    def setup_control(self):
        self.ui.DrawContour_1_1.setEnabled(False)
        self.ui.CountRings1_2.setEnabled(False)
        self.ui.FindCorners_2_1.setEnabled(False)
        self.ui.FindIntrinsic_2_2.setEnabled(False)
        self.ui.Find_Extrinsic_2_3.setEnabled(False)
        self.ui.Find_Distortion_2_4.setEnabled(False)
        self.ui.ShowResult_2_5.setEnabled(False)
        self.ui.ShowWordsOnBoard_3_1.setEnabled(False)
        self.ui.ShowWordsVertically_3_2.setEnabled(False)
        self.ui.SteroDisparityMap_4_1.setEnabled(False)
        self.ui.DrawContour_1_1.clicked.connect(self.DrawContour)
        self.ui.CountRings1_2.clicked.connect(self.CountRings)
        self.ui.FindCorners_2_1.clicked.connect(self.FindCorners)
        self.ui.FindIntrinsic_2_2.clicked.connect(self.FindIntrinsic)
        self.ui.Find_Extrinsic_2_3.clicked.connect(self.FindExtrinsic)
        self.ui.Find_Distortion_2_4.clicked.connect(self.FindDistortion)
        self.ui.ShowResult_2_5.clicked.connect(self.ShowResult)
        self.ui.ShowWordsOnBoard_3_1.clicked.connect(self.ShowWordsOnBoard)
        self.ui.ShowWordsVertically_3_2.clicked.connect(self.ShowWordsVertically)
        self.ui.SteroDisparityMap_4_1.clicked.connect(self.SteroDisparityMap)
        self.ui.LoadImageL.clicked.connect(self.OpenImageL)
        self.ui.LoadImageR.clicked.connect(self.OpenImageR) 
        self.ui.LoadFolder.clicked.connect(self.OpenFolder)
        
    # 1.1 DrawContour 
    def DrawContour(self):
        global filepathL, filepathR
        imgR = cv_imread(filepathR)
        imgL = cv_imread(filepathL)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        retR,threshR = cv2.threshold(grayR,127,255,cv2.THRESH_BINARY)
        retL,threshL = cv2.threshold(grayL,127,255,cv2.THRESH_BINARY)
        contoursL,_  = cv2.findContours(threshL,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contoursR,_ = cv2.findContours(threshR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('imageshowL',imgL)
        cv2.imshow('imageshowR',imgR)
        imageL = cv2.drawContours(imgL.copy(),contoursL,-1,(0,255,0),3)
        imageR = cv2.drawContours(imgR.copy(),contoursR,-1,(0,255,0),3)
        cv2.imshow('drawingL',imageL)
        cv2.imshow('drawingR',imageR)
        cv2.waitKey(0)

    # 1.2 Count Rings
    #CountRings = CountRings
    #count how many rings in the image
    def CountRings(self):
        global filepathL, filepathR
        imgR = cv_imread(filepathR)
        imgL = cv_imread(filepathL)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        retR,threshR = cv2.threshold(grayR,127,255,cv2.THRESH_BINARY)
        retL,threshL = cv2.threshold(grayL,127,255,cv2.THRESH_BINARY)
        contoursL,_  = cv2.findContours(threshL,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contoursR,_ = cv2.findContours(threshR,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        self.ui.Rings_In_Img1.setText("There are " + str(int(len(contoursL)/2)) + " rings in the left image")
        self.ui.Rings_In_Img2.setText("There are " + str(int(len(contoursR)/2)) + " rings in the right image")
    
    # 2.1 Find Corners
    def FindCorners(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane

        global all_file_name
        image_result = []
        i=0
        for img in all_file_name:
            i = i +1
            # Load the image
            image = cv_imread(img)            
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
           
            nx=11
            ny=8
            retu, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if retu == True:  # If found, draw corners
            # Draw and display the corners
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(image, (nx, ny), corners, retu)
                image_result.append(image)
        # show the result
        #Click button “2.1” to show each picture 0.5 seconds.
        for img in image_result:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.3)
            cv2.imshow('img',img)
            cv2.waitKey(500)
        print(i)
        cv2.destroyAllWindows()
        global ret, mtx, dist, rvecs, tvecs
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
        
    # 2.2 Find Intrinsic
    def FindIntrinsic(slef):        
        global mtx
        print("Intrinsic\n",mtx)

    # 2.3 Find Extrinsic
    def FindExtrinsic(self):
        global mtx, dist, rvecs, tvecs
        value =  self.ui.spinBox.value()
        R = cv2.Rodrigues(rvecs[value-1])
        ext = np.hstack((R[0], tvecs[value-1]))
        print("Extrinsic:\n",ext)

    # 2.4 Find Distortion
    def FindDistortion(self):
        global dist
        print("Distortion:\n", dist) # Find the Extrinsic Matrix of the camera

    # 2.5 Show Result
    def ShowResult(self):

        global ret, mtx, dist, rvecs, tvecs
        global all_file_name
        image_result = []
        for image in all_file_name:
            img = cv2.imread(image)
            #img = cv_imread(image)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h)) 
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # resize the dst smaller 
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.15)
            # resize the dst's photo size to img's photo size
            h,w = img.shape[:2] 
            dst = cv2.resize(dst, (w,h), interpolation=cv2.INTER_AREA)
            # merge the dst and img
            vis = np.concatenate((img,dst), axis=1)
            image_result.append(vis)
        for img in image_result:
            cv2.imshow('img',img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    # 3.1 Show Words On Board
    def ShowWordsOnBoard(self):
        text = self.ui.textEdit.toPlainText()
        text = text.upper()
        for i in range(1,6):
            x = 7
            y = 5
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane
            global folder_path
            filename = folder_path + "/" + str(i) + ".bmp"
            #image = cv2.imread(filename)
            image = cv_imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            numchar = 1
            for char in text:
                fs = cv2.FileStorage('alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(char).mat() 
                ch = ch.reshape(-1, 3)
            
                for j in range(ch.shape[0]):
                    ch[j][0] += x
                    ch[j][1] += y 
                pyramids = np.array(ch,dtype=numpy.float32)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
                twodpro, jac = cv2.projectPoints(pyramids, rvecs[0], tvecs[0], mtx, dist)
                for j in range(0, ch.shape[0], 2):
                    image = cv2.line(image, (int(twodpro[j][0][0]), int(twodpro[j][0][1])), (int(twodpro[j+1][0][0]), int(twodpro[j+1][0][1])), (0, 0, 255), 2)                
                x -= 3
                numchar += 1
                if numchar == 4:
                    x = 7
                    y = 2
            image = cv2.resize(image,(512,512))
            cv2.imshow('img',image)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()
    # 3.2 Show Words Vertically
    def ShowWordsVertically(self):
        text = self.ui.textEdit.toPlainText()
        text = text.upper()
        for i in range(1,6):
            x = 7
            y = 5
            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane
            global folder_path
            filename = folder_path + "/" + str(i) + ".bmp"
            #image = cv2.imread(filename)
            image = cv_imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            objp = np.zeros((11 * 8, 3), np.float32)
            objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
            numchar = 1
            for char in text:
                fs = cv2.FileStorage('alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
                ch = fs.getNode(char).mat() 
                ch = ch.reshape(-1, 3)
            
                for j in range(ch.shape[0]):
                    ch[j][0] += x
                    ch[j][1] += y 
                pyramids = np.array(ch,dtype=numpy.float32)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
                twodpro, jac = cv2.projectPoints(pyramids, rvecs[0], tvecs[0], mtx, dist)
                for j in range(0, ch.shape[0], 2):
                    image = cv2.line(image, (int(twodpro[j][0][0]), int(twodpro[j][0][1])), (int(twodpro[j+1][0][0]), int(twodpro[j+1][0][1])), (0, 0, 255), 2)                
        
                x -= 3
                numchar += 1
                if numchar == 4:
                    x = 7
                    y = 2
            image = cv2.resize(image,(512,512))
            cv2.imshow('img',image)
            cv2.waitKey(1000)
    # 4.1 Stero Disparity Map
    def SteroDisparityMap(self):
        global filepathL
        global filepathR
        self.imgL = cv2.imread(filepathL)
        self.imgR = cv2.imread(filepathR)
        grayL = cv2.imread(filepathL, cv2.IMREAD_GRAYSCALE)
        grayR = cv2.imread(filepathR, cv2.IMREAD_GRAYSCALE)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(grayL, grayR)
        plt.imshow(disparity,'gray')
        plt.show()

        disp_norm = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        def draw_match(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                x = x * 4
                y = y * 4
                #if x > disparity.shape[1]:
                #    return
                disp = int(disparity[y][x] / 16)
                # if disp <= 0:
                #  return
                print(x, y, disp)
                imgL = self.imgL.copy()
                imgR = self.imgR.copy()
                point = (x - disp, y)
                imgR = cv2.circle(imgR, point, 10, (0, 255, 0), -1)
                horizontal = np.hstack((imgL, imgR))
                horizontal = cv2.resize(horizontal, (horizontal.shape[1] // 4, horizontal.shape[0] // 4))
                cv2.imshow('match', horizontal)
        horizontal = np.hstack((self.imgL, self.imgR))
        horizontal = cv2.resize(horizontal, (horizontal.shape[1] // 4, horizontal.shape[0] // 4))
        cv2.namedWindow('match')
        cv2.setMouseCallback('match', draw_match)
        cv2.imshow('match', horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Open Folder 讀取資料夾內的所有圖片
    def OpenFolder(self):
        global all_file_name
        global folder_path
        all_file_name = []
        # 顯示選擇資料夾的對話框
        folder_path = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder_path == "":
            return
        # 顯示資料夾的路徑
        self.ui.LoadFolderLabel.setText(folder_path)
        # 存取資料夾內所有的檔案名稱
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".bmp") or file.endswith(".png"):
                    all_file_name.append(os.path.join(root, file))

        print(all_file_name)    
        self.ui.FindCorners_2_1.setEnabled(True)
        self.ui.FindIntrinsic_2_2.setEnabled(True)
        self.ui.Find_Extrinsic_2_3.setEnabled(True)
        self.ui.Find_Distortion_2_4.setEnabled(True)
        self.ui.ShowResult_2_5.setEnabled(True)
        self.ui.ShowWordsOnBoard_3_1.setEnabled(True)
        self.ui.ShowWordsVertically_3_2.setEnabled(True)
    # Open Image R
    def OpenImageR(self):
        global filepathR
        filepathR , ok = QFileDialog.getOpenFileName(self,"Open file","./Dataset_OpenCvDl_Hw2/")  
        if ok:
            filenameR = os.path.basename(filepathR)
            filenameR = filenameR.split('.')[0]
            print(filenameR)
            self.ui.LoadImageRLabel.setText(filenameR)
            self.ui.DrawContour_1_1.setEnabled(True)
            self.ui.CountRings1_2.setEnabled(True)
            self.ui.FindCorners_2_1.setEnabled(True)
            self.ui.FindIntrinsic_2_2.setEnabled(True)
            self.ui.Find_Extrinsic_2_3.setEnabled(True)
            self.ui.Find_Distortion_2_4.setEnabled(True)
            self.ui.ShowResult_2_5.setEnabled(True)
            self.ui.SteroDisparityMap_4_1.setEnabled(True)

    # Open Image L
    def OpenImageL(self):
        global filepathL
        filepathL , ok = QFileDialog.getOpenFileName(self,"Open file","./Dataset_OpenCvDl_Hw2/")  
        if ok:
           
            filenameL = os.path.basename(filepathL)
            filenameL = filenameL.split('.')[0]
            print(filenameL)
            self.ui.LoadImageLLabel.setText(filenameL)
            self.ui.DrawContour_1_1.setEnabled(True)
            self.ui.CountRings1_2.setEnabled(True)
            self.ui.FindCorners_2_1.setEnabled(True)
            self.ui.FindIntrinsic_2_2.setEnabled(True)
            self.ui.Find_Extrinsic_2_3.setEnabled(True)
            self.ui.Find_Distortion_2_4.setEnabled(True)
            self.ui.ShowResult_2_5.setEnabled(True)
            self.ui.SteroDisparityMap_4_1.setEnabled(True)
            
        