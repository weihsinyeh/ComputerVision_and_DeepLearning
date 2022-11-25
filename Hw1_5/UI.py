# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(660, 525)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 40, 179, 205))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.LoadImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.LoadImage.setObjectName("LoadImage")
        self.verticalLayout.addWidget(self.LoadImage)
        self.ShowTrainImages = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ShowTrainImages.setObjectName("ShowTrainImages")
        self.verticalLayout.addWidget(self.ShowTrainImages)
        self.ShowModelStructure = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ShowModelStructure.setObjectName("ShowModelStructure")
        self.verticalLayout.addWidget(self.ShowModelStructure)
        self.ShowDataAugumentation = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ShowDataAugumentation.setObjectName("ShowDataAugumentation")
        self.verticalLayout.addWidget(self.ShowDataAugumentation)
        self.ShowAccuracyandLoss = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ShowAccuracyandLoss.setObjectName("ShowAccuracyandLoss")
        self.verticalLayout.addWidget(self.ShowAccuracyandLoss)
        self.Inference = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.Inference.setObjectName("Inference")
        self.verticalLayout.addWidget(self.Inference)
        self.EdgeDetection = QtWidgets.QLabel(Dialog)
        self.EdgeDetection.setGeometry(QtCore.QRect(30, 0, 141, 31))
        self.EdgeDetection.setObjectName("EdgeDetection")
        self.Image = QtWidgets.QLabel(Dialog)
        self.Image.setGeometry(QtCore.QRect(270, 140, 321, 291))
        self.Image.setObjectName("Image")
        self.ImageTitle = QtWidgets.QLabel(Dialog)
        self.ImageTitle.setGeometry(QtCore.QRect(270, 0, 331, 81))
        self.ImageTitle.setObjectName("ImageTitle")
        self.ImageTitle_2 = QtWidgets.QLabel(Dialog)
        self.ImageTitle_2.setGeometry(QtCore.QRect(270, 50, 331, 81))
        self.ImageTitle_2.setObjectName("ImageTitle_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.LoadImage.setText(_translate("Dialog", "Load Image"))
        self.ShowTrainImages.setText(_translate("Dialog", "1. Show Train Images"))
        self.ShowModelStructure.setText(_translate("Dialog", "2. Show Model Structure"))
        self.ShowDataAugumentation.setText(_translate("Dialog", "3. Show Data Augumentation"))
        self.ShowAccuracyandLoss.setText(_translate("Dialog", "4. Show Accuracy and Loss"))
        self.Inference.setText(_translate("Dialog", "5. Inference"))
        self.EdgeDetection.setText(_translate("Dialog", "5. Resnet101 Test"))
        self.Image.setText(_translate("Dialog", "Image"))
        self.ImageTitle.setText(_translate("Dialog", "ImageTitle"))
        self.ImageTitle_2.setText(_translate("Dialog", "ImageTitle"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

