"""
CNNVisTool: Convolutional Neural Network Visualization Tool is an open source software written in Python 3 that allows users to load different trained convolutional neural network model for classifying images. The software is designed to provide visualization chart to understand the output of the trained model, and to save the predictions for further analysis.

This software is developed as part of the SPECTORS Project, Dutch-German cooperation project is funded by INTERREG V-A Deutschland-Nederland (project number 143081).

Version 2.2.0
NOTE: remember to change [self.appVersion] when version is changed

History
    [j: version targeted three vegetation species/type namely Jaocoabea, Rumex and Others]
    [r: version targeted two vegetation species/type namely Rumex and Others]

    v1.0.0j (20.04.2018): 20180418-Salmorth-60-0.25-1e-05-VGG16-17epoch_run-2    68.67%
    v1.0.0r (21.10.2018): 20181020-Rumex_Others-100-0.2-1e-05-VGG16-20epoch_run-1	96.20% (Rumex and Others only)
    v1.0.1j (25.04.2018): 20180424-Salmorth-100-0.25-1e-05-VGG16-13epoch_run-2   72.83%
    v1.1.0j (22.06.2018): Allow user to save the predictions in .txt
    v1.1.1j (25.06.2018): Display version on window title; Updated header for text file
    v2.0.0j (25.01.2019): Allow selection of trained model (automatically select correct architecture in background)
    v2.0.0r (05.06.2019): Allow selection of trained model (automatically select correct architecture in background)
    v2.0.1r (02.10.2019): Save the model name in the generated text file header
    v2.1.0  (13.12.2019): (1) No pre-loading of model during initialization, only display the GUI
                          (2) When select the model, the text file with the class list will automatically be loaded
                          (3) Application is now possible to be used for different trained VGG16 model despite the number of classes
                              when provided with appropriate file naming
    v2.1.1  (13.12.2019): Renmae the application from ImageClassifier to CNNVisTool (CNN Visulization Tool)
    v2.1.2  (15.12.2019): Embedded .ico into single executable compiled by PyInstaller 3.4
    v2.2.0  (18.12.2019): (1) Modified error messages for failed to load images and model.
                          (2) Fixed issue raised when trained model is not saved under "trained_model/"
                          (3) Fixed issue raised when saving prediction when failed to load selected model
                          (4) Commented out the archiecture for ResNet32 cause it is not ready
"""
import sys, os, matplotlib, time
import numpy as np
from datetime import datetime, timedelta

matplotlib.use("Qt5Agg")

from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.appVersion = "2.2.0"
        self.imgDir = ""
        self.imgList = []

        # dummy variables for trained model
        self.MODEL_NAME = "20181020-Rumex_Others-100-0.2-1e-05-VGG16-20epoch_run-1.tfl"
        self.IMG_SIZE = 100
        self.NUM_CHANNEL = 3
        self.NUM_CLASS = 2
        self.LR = 1e-5
        self.NAME_NETWORK = "VGG16"
        self.classList = ["Others", "Rumex"]
        self.classCounter = [0, 0]

        self.bool_networkSetup = True
        self.bool_modelLoad = True

        MainWindow.setWindowTitle('CNNVisTool (v{})'.format(self.appVersion))
        MainWindow.setFixedSize(1200, 650)

        # set window icon for single executable compiled with PyInstaller 3.4
        if hasattr(sys, '_MEIPASS'):
            MainWindow.setWindowIcon(QIcon(os.path.join(sys._MEIPASS, "img/CNNVisTool.ico")))

        # set window icon for simply executing .py file
        else:
            MainWindow.setWindowIcon(QIcon("img/CNNVisTool.ico"))
        self.status = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.status)

        # self.setupNN()
        self.run(MainWindow)

        message = "Initialization completed."
        if((self.bool_networkSetup == False) or (self.bool_modelLoad == False)): message = "Initialization failed, please check the log."
        print(message)
        self.status.showMessage(message)

        MainWindow.show()

    def run(self, MainWindow):
        widget_main = QWidget(MainWindow)

        '''Header - auto select text file for list of classes'''
        widget_header_02 = QWidget(widget_main)
        widget_header_02.setGeometry(QRect(0, 0, 1200, 65))
        header_02 = QFormLayout(widget_header_02)
        header_02.setContentsMargins(10, 10, 10, 10)

        self.label_seletedClassList = QLabel('Waiting for a model to be selected...', widget_header_02)

        header_02.setWidget(0, QFormLayout.FieldRole, self.label_seletedClassList)


        '''Header - select model'''
        widget_header_01 = QWidget(widget_main)
        widget_header_01.setGeometry(QRect(0, 30, 1200, 65))
        header_01 = QFormLayout(widget_header_01)
        header_01.setContentsMargins(10, 10, 10, 10)

        self.label_seletedModel = QLabel("...", widget_header_01)

        header_01.setWidget(0, QFormLayout.FieldRole, self.label_seletedModel)

        btn_selectModel = QPushButton("Select model", widget_header_01)
        btn_selectModel.setToolTip("Select the trained model for predicting new images.")
        btn_selectModel.resize(btn_selectModel.sizeHint())
        btn_selectModel.clicked.connect(self.selectModelFile)

        header_01.setWidget(0, QFormLayout.LabelRole, btn_selectModel)


        '''Header - select image folder'''
        widget_header = QWidget(widget_main)
        widget_header.setGeometry(QRect(0, 65, 1200, 65))
        header = QFormLayout(widget_header)
        header.setContentsMargins(10, 10, 10, 10)

        self.label_seletedFolder = QLabel("...", widget_header)

        header.setWidget(0, QFormLayout.FieldRole, self.label_seletedFolder)

        btn_selectFolder = QPushButton("Select folder", widget_header)
        btn_selectFolder.setToolTip("Select the folder containing the images.")
        btn_selectFolder.resize(btn_selectFolder.sizeHint())
        btn_selectFolder.clicked.connect(self.selectImageDirectory)

        header.setWidget(0, QFormLayout.LabelRole, btn_selectFolder)


        '''Image Preview'''
        widget_imagePreview = QWidget(widget_main)
        widget_imagePreview.setGeometry(QRect(0, 110, 600, 510))
        hbox_imagePreview = QHBoxLayout(widget_imagePreview)
        hbox_imagePreview.setContentsMargins(10, 10, 10, 10)

        gb_imagePreview = QGroupBox("Image Preview", widget_imagePreview)

        self.label_imgName = QLabel(gb_imagePreview)
        self.label_imgName.resize(500, 25)
        self.label_imgName.move(10, 455)
        self.label_imgName.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)

        self.label_imgNum = QLabel(gb_imagePreview)
        self.label_imgNum.resize(100, 25)
        self.label_imgNum.move(470, 455)
        self.label_imgNum.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.label_imagePreview = QLabel(gb_imagePreview)
        self.label_imagePreview.setGeometry(QRect(10, 30, 560, 410))
        self.label_imagePreview.setAlignment(Qt.AlignCenter)

        hbox_imagePreview.addWidget(gb_imagePreview)


        '''Results'''
        widget_prediction = QWidget(widget_main)
        widget_prediction.setGeometry(QRect(600, 110, 600, 510))
        hbox_prediction = QHBoxLayout(widget_prediction)
        hbox_prediction.setContentsMargins(10, 10, 10, 10)

        gb_prediction = QGroupBox("Predictions", widget_prediction)

        btn_previous = QPushButton("Previous", gb_prediction)
        btn_previous.setToolTip("Display previous image in selected folder")
        btn_previous.resize(btn_previous.sizeHint())
        btn_previous.move(340, 450)
        btn_previous.clicked.connect(self.displayPreviousImage)

        btn_next = QPushButton("Next", gb_prediction)
        btn_next.setToolTip("Display next image in selected folder")
        btn_next.resize(btn_next.sizeHint())
        btn_next.move(460, 450)
        btn_next.clicked.connect(self.displayNextImage)

        btn_savePrediction = QPushButton("Save Predictions", gb_prediction)
        btn_savePrediction.setToolTip("Save the predictions into a text file (.txt)")
        btn_savePrediction.resize(btn_savePrediction.sizeHint())
        btn_savePrediction.move(25, 450)
        btn_savePrediction.clicked.connect(self.savePredictionToFile)

        self.gviewer_prediction = QGraphicsView(gb_prediction)
        self.gviewer_prediction.setGeometry(QRect(10, 30, 560, 410))
        self.gviewer_prediction.setStyleSheet("background:transparent; border:0px;")

        self.gscene_prediction = QGraphicsScene(self.gviewer_prediction)
        self.gscene_prediction.setSceneRect(0, 0, 550, 400)

        hbox_prediction.addWidget(gb_prediction)

        MainWindow.setCentralWidget(widget_main)


    def selectModelFile(self):
        cwd = str(os.getcwd())
        modelFile = QFileDialog.getOpenFileName(None, 'Select File', '{}/trained_model'.format(cwd), 'TFLearn Files (*.data-00000-of-00001)')

        self.MODEL_DIR = os.path.dirname(modelFile[0])
        self.MODEL_NAME = os.path.basename(modelFile[0].split('.data-00000-of-00001')[0])

        try:
            # get the parameters from model name
            m = self.MODEL_NAME.split(sep='-')

            if (m[-3] == 'VGG16'):
                self.IMG_SIZE, self.LR, self.NAME_NETWORK = int(m[3]), m[5], m[-3]
                if (self.LR == '1e'): self.LR = '{}-{}'.format(m[5], m[6])
                print('self.IMG_SIZE: {}, self.LR: {}, self.NAME_NETWORK: {}'.format(self.IMG_SIZE, self.LR, self.NAME_NETWORK))

            # load selected model
            message = "Loading selected model..."
            print(message)
            self.status.showMessage(message)

            # load class list from text file which has the same name as the model file
            classFile = open('{}.txt'.format(modelFile[0].split('.tfl')[0]), 'r')

            first_line = classFile.readline()
            self.classList = first_line.split(",")
            self.NUM_CLASS = len(self.classList)

            print("Selected class list loaded")
            classFile.close()

            # update class list displayed
            self.label_seletedClassList.setText('{} classes: {}'.format(str(self.NUM_CLASS), str(self.classList)))

            # setup neural network according to the parameters
            self.setupNN()

            if (len(self.imgList) > 0):
                self.predictClass()

            # update model name displayed
            self.label_seletedModel.setText(str(self.MODEL_NAME))

        except:
            if (modelFile[0] == ''):
                message = "No model has been selected. The last trained model (if exists) is maintained."
                print(message)
                self.status.showMessage(message)

            else:
                # if no file is selected or if the dialog is closed, display the message on python console and statusbar
                message = "Failed to load selected model. Please check the file naming of the selected .tfl file. The last trained model (if exists) is maintained."
                print(message)
                self.status.showMessage(message)

            pass


    def selectImageDirectory(self):
        cwd = str(os.getcwd())
        self.imgDir = QFileDialog.getExistingDirectory(None, "Select Folder", cwd, QFileDialog.ShowDirsOnly)

        try:
            # get the size of image set in the directory
            self.imgList = []
            imageExtenstions = ["jpg", "png"]
            tmp_imgList = [fn for fn in os.listdir(self.imgDir)
                         if any(fn.lower().endswith(ext) for ext in imageExtenstions)]        # used wihen there is no subdirectories
            self.imgList += tmp_imgList

            imgListSize = len(self.imgList)

            # display the name of folder in python console, statusbar and label
            print('{} is selected.'.format(self.imgDir))
            self.label_seletedFolder.setText(str(self.imgDir))
            self.status.showMessage('{} is selected.'.format(self.imgDir))

            # print the number of images and the image list in the python console
            print('{} images found in {}'.format(imgListSize, self.imgDir))
            print(self.imgList)

            # display image on QGraphicsScene
            self.img = os.path.join(self.imgDir, self.imgList[0])

            self.displayImage()

            # display the name of the image on the label and the "progress counter" on the other label
            self.imgName = self.imgList[0]
            self.label_imgName.setText(str(self.imgName))
            self.label_imgNum.setText('{}/{}'.format(1, len(self.imgList)))

            print(self.img)

            self.predictClass()

        except:
            # if there is no image in the selected folder
            if (len(self.imgList) == 0 and os.path.isdir(self.imgDir)):
                message = 'No images found in the selected folder {}'.format(self.imgDir)
                # print(message)
                self.status.showMessage(message)

            # if no folder is selected or if the dialog is closed, display the message on python console and statusbar
            else:
                message = "No directory is selected. Failed to load images."
                # print(message)
                self.status.showMessage(message)

            # reset the label of selected folder, image name and "progress counter"
            self.label_seletedFolder.setText("...")
            self.imgName = ""
            self.label_imgName.setText("")
            self.label_imgNum.setText("")

            # clear the QGraphicsView
            self.gscene_prediction.clear()
            self.label_imagePreview.clear()

            pass


    def savePredictionToFile(self):
        self.status.showMessage("")

        if (len(self.classCounter) > 0):
            self.classCounter.clear()

        try:
            if(len(self.imgList) == 0):
                message = "Please select a folder to proceed."
                print(message)
                self.status.showMessage(message)

            else:
                message = "Saving...Please wait..."
                print(message)
                self.status.showMessage(message)

                dateString = datetime.strftime(datetime.now(), "%Y%m%d")
                timeString = datetime.strftime(datetime.now(), "%H%M%S")

                fileName = '{}_{}_predictions.txt'.format(dateString, timeString)
                filePath = os.path.join(self.imgDir, fileName)

                self.classCounter = [0] * len(self.classList)

                startSaveTime = time.time()     # start timer

                file = open(filePath, "w+")
                file.write('======================================\nFile created on: {}.{}.{} {}:{}:{}\nApplication version: {} (For more information, please refer to the user manual.)\nModel name: {}\nImage directory: {}\n\nData description:\n1st column: image name\n2nd column: an array specifying the prediction percentage of each class {}\n3rd column: predicted class, i.e. the class with the highest percentage\n======================================\n'.format(dateString[6:], dateString[4:6], dateString[:4], timeString[:2], timeString[2:4], timeString[4:], self.appVersion, self.label_seletedModel.text(), self.imgDir, self.classList))

                for i in range(len(self.imgList)):
                    self.img = os.path.join(self.imgDir, self.imgList[i])
                    print('{}: {}'.format(i+1, self.img))

                    self.predictClass()

                    self.classCounter[self.predictionIndex] = self.classCounter[self.predictionIndex] + 1

                    file.write('{}\t{}\t{}\n'.format(self.imgList[i], self.model_out[0]*100, self.classPredicted))

                file.write('======================================\nTotal number of images: {}\n'.format(len(self.imgList)))

                for classIndex in range(len(self.classCounter)):
                    file.write('Number of predicted {}: {}\n'.format(self.classList[classIndex], self.classCounter[classIndex]))

                file.close()

                endSaveTime = time.time()     # stop timer
                print('Save prediction process time: {}'.format(timedelta(seconds=(endSaveTime - startSaveTime))))

                message = 'Predictions saved at {}.'.format(filePath)
                print(message)
                self.status.showMessage(message)

        except:
            pass


    def displayNextImage(self):
        self.status.showMessage("")

        try:
            imgIndex = self.imgList.index(self.imgName)

            if (imgIndex < len(self.imgList)-1):
                self.img = os.path.join(self.imgDir, self.imgList[imgIndex+1])

                self.displayImage()

                self.imgName = self.imgList[imgIndex+1]
                self.label_imgName.setText(str(self.imgName))
                self.label_imgNum.setText('{}/{}'.format(imgIndex+2, len(self.imgList)))

                print(self.img)

                self.predictClass()

            else:
                message = "This is the last image in this folder."
                print(message)
                self.status.showMessage(message)

        except:
            message = "Please select a folder to proceed."
            print(message)
            self.status.showMessage(message)

            pass


    def displayPreviousImage(self):
        self.status.showMessage("")

        try:
            imgIndex = self.imgList.index(self.imgName)

            if (imgIndex > 0):
                self.img = os.path.join(self.imgDir, self.imgList[imgIndex-1])

                self.displayImage()

                self.imgName = self.imgList[imgIndex-1]
                self.label_imgName.setText(str(self.imgName))
                self.label_imgNum.setText('{}/{}'.format(imgIndex, len(self.imgList)))

                print(self.img)

                self.predictClass()

            else:
                message = "This is the first image in this folder."
                print(message)
                self.status.showMessage(message)

        except:
            message = "Please select a folder to proceed."
            print(message)
            self.status.showMessage(message)

            pass


    def displayImage(self):
        self.label_imagePreview.clear()
        self.label_imagePreview.setPixmap(QPixmap(self.img).scaled(self.label_imagePreview.size(), Qt.KeepAspectRatio))

        print(QPixmap(self.img).size())


    # setup neural network structure according to NAME_NETWORK
    def setupNN(self):
        try:
            import tensorflow as tf
            tf.reset_default_graph()

            import tflearn
            from tflearn.layers.conv import conv_2d, max_pool_2d
            from tflearn.layers.core import input_data, dropout, fully_connected
            from tflearn.layers.estimator import regression

            # fix random seed for reproducibility
            np.random.seed(7070)
            tflearn.config.init_graph(7070)

            message = '{} is used. Setting up {}...'.format(self.NAME_NETWORK, self.NAME_NETWORK)
            print(message)
            self.status.showMessage(message)

            # define the network structure
            convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNEL], name="input")

            # VGG16 structure
            if (self.NAME_NETWORK == "VGG16"):
                convnet = conv_2d(convnet, 64, 3, activation='relu')
                convnet = conv_2d(convnet, 64, 3, activation='relu')
                convnet = max_pool_2d(convnet, 2, strides=2)

                convnet = conv_2d(convnet, 128, 3, activation='relu')
                convnet = conv_2d(convnet, 128, 3, activation='relu')
                convnet = max_pool_2d(convnet, 2, strides=2)

                convnet = conv_2d(convnet, 256, 3, activation='relu')
                convnet = conv_2d(convnet, 256, 3, activation='relu')
                convnet = conv_2d(convnet, 256, 3, activation='relu')
                convnet = max_pool_2d(convnet, 2, strides=2)

                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = max_pool_2d(convnet, 2, strides=2)

                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = conv_2d(convnet, 512, 3, activation='relu')
                convnet = max_pool_2d(convnet, 2, strides=2)

                convnet = fully_connected(convnet, 4096, activation='relu')
                convnet = dropout(convnet, 0.5)
                convnet = fully_connected(convnet, 4096, activation='relu')
                convnet = dropout(convnet, 0.5)
                convnet = fully_connected(convnet, self.NUM_CLASS, restore=False, activation='softmax')

                convnet = regression(convnet, optimizer="adam", learning_rate=self.LR, loss="categorical_crossentropy", shuffle_batches=True, name="targets")

            # # For archiecture other than VGG16
            # # ResNet structure
            # # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
            # elif (NAME_NETWORK == "ResNet32"):
            #     n = 5
            #
            #     convnet = conv_2d(convnet, 16, 3, regularizer='L2', weight_decay=0.0001)
            #     convnet = residual_block(convnet, n, 16)
            #     convnet = residual_block(convnet, 1, 32, downsample=True)
            #     convnet = residual_block(convnet, n-1, 32)
            #     convnet = residual_block(convnet, 1, 64, downsample=True)
            #     convnet = residual_block(convnet, n-1, 64)
            #     convnet = batch_normalization(convnet)
            #     convnet = tflearn.activation(convnet, 'relu')
            #     convnet = global_avg_pool(convnet)
            #     # Regression
            #     convnet = fully_connected(convnet, NUM_CLASS, restore=False, activation='softmax')
            #     # mom = tflearn.Momentum(learning_rate=0.0000005, lr_decay=0.9, decay_step=1000, staircase=True)
            #     mom = tflearn.Momentum(learning_rate=0.1, lr_decay=0.1, decay_step=1000, staircase=True)
            #     convnet = regression(convnet, optimizer=mom, loss='categorical_crossentropy', shuffle_batches=False, name="targets")


            self.model = tflearn.DNN(convnet)

            message = '{} is setup. Loading trained model...'.format(self.NAME_NETWORK)
            print(message)
            self.status.showMessage(message)

            self.loadModel()

        except:
            self.bool_networkSetup = False
            message = '{} failed to setup.'.format(self.NAME_NETWORK)
            print(message)
            self.status.showMessage(message)

            pass


    # load the trained model
    def loadModel(self):
        try:
            self.model.load(os.path.join(self.MODEL_DIR, self.MODEL_NAME))

            message = "Trained model is loaded."
            print(message)
            self.status.showMessage(message)

        except:
            self.bool_modelLoad = False
            message = '{} failed to load.'.format(self.MODEL_NAME)
            print(message)
            self.status.showMessage(message)

            pass


    # feed the displayed image to the loaded model
    def predictClass(self):
        try:
            # conver the image into array
            image = Image.open(self.img)    # RGB
            image = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.NEAREST)
            image.load()
            image = np.asarray(image, dtype="uint8")

            # feed the image array to the trained model
            self.model_out = self.model.predict(image.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, self.NUM_CHANNEL))

            # get the class name with the highest probability
            self.predictionIndex = np.argmax(self.model_out)
            self.classPredicted = self.classList[self.predictionIndex]

            print('modelOut: {} => {}'.format(self.model_out[0]*100, self.classPredicted))

            # plot bar chart
            plt = Figure_Canvas(self.classList, self.model_out, self.NAME_NETWORK)

            self.gscene_prediction.clear()
            self.gscene_prediction.addWidget(plt)
            self.gviewer_prediction.setScene(self.gscene_prediction)

        except:
            self.gscene_prediction.clear()
            message = "Prediction failed."
            print(message)
            self.status.showMessage(message)

            pass

class Figure_Canvas(FigureCanvas):
    def __init__(self, classList, modelOut, networkName, parent=None, width=5.5, height=4, dpi=100):
        self.classList = classList
        self.modelOut = modelOut
        self.networkName = networkName

        # plot bar chart using Figure from matplotlib, not from pyplot
        fig = Figure(figsize=(width, height), dpi=100)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.axes = fig.add_subplot(111)

        self.plotPrediction()

    def plotPrediction(self):
        x_pos = np.arange(len(self.classList))
        bar_width = 0.5

        self.axes.bar(x_pos, self.modelOut[0]*100, bar_width, color='g', align='center')
        self.axes.set_xticks(x_pos)
        self.axes.set_xticklabels(self.classList)
        self.axes.set_xlabel("Classes")
        self.axes.set_ylabel("Percentage [%]")
        self.axes.set_yticks(np.arange(0, 110, 10))
        self.axes.set_title('Prediction using {}'.format(self.networkName))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    sys.exit(app.exec_())
