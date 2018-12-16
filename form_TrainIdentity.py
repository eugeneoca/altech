# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form_TrainIdentity.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(356, 187)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_startTraining = QtWidgets.QPushButton(self.centralwidget)
        self.btn_startTraining.setObjectName("btn_startTraining")
        self.gridLayout.addWidget(self.btn_startTraining, 2, 0, 1, 1)
        self.progress_Training = QtWidgets.QProgressBar(self.centralwidget)
        self.progress_Training.setProperty("value", 0)
        self.progress_Training.setObjectName("progress_Training")
        self.gridLayout.addWidget(self.progress_Training, 1, 0, 1, 1)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.formLayout = QtWidgets.QFormLayout(self.frame)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.txt_TotalIdentities = QtWidgets.QLineEdit(self.frame)
        self.txt_TotalIdentities.setObjectName("txt_TotalIdentities")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_TotalIdentities)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.txt_TrainingProcess = QtWidgets.QLineEdit(self.frame)
        self.txt_TrainingProcess.setObjectName("txt_TrainingProcess")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.txt_TrainingProcess)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 356, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Train Datasets"))
        self.btn_startTraining.setText(_translate("MainWindow", "Start Training"))
        self.label.setText(_translate("MainWindow", "Total Identities:"))
        self.label_2.setText(_translate("MainWindow", "Process:"))

