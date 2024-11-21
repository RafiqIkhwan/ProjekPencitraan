import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tampilan import Ui_MainWindow
from scipy import ndimage
from PyQt5.QtWidgets import QButtonGroup

class ImageProcessor(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setup_connections()
        self.original_image = None
        self.processed_image = None
        
        self.camera = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.biner)
        self.radio_group.addButton(self.negatif)
        self.radio_group.addButton(self.Brigth)
        self.radio_group.addButton(self.Flip)
        self.radio_group.addButton(self.Sharp)
        self.radio_group.addButton(self.blur)
        self.radio_group.addButton(self.gray)
        self.radio_group.addButton(self.edge)
        
    def setup_connections(self):
        self.actionOpen.triggered.connect(self.openImage)
        self.actionSave.triggered.connect(self.saveImage)
        self.actionBuka_Kamera_2.triggered.connect(self.start_camera)
        self.actionTutup_Kamera_2.triggered.connect(self.stop_camera)
        
        self.biner.toggled.connect(lambda: self.menu.setCurrentWidget(self.binarMenu))
        self.negatif.toggled.connect(lambda: self.menu.setCurrentWidget(self.negatifMenu))
        self.Brigth.toggled.connect(lambda: self.menu.setCurrentWidget(self.brigthMenu))
        self.Flip.toggled.connect(lambda: self.menu.setCurrentWidget(self.flipMenu))
        self.Sharp.toggled.connect(lambda: self.menu.setCurrentWidget(self.sharpMenu))
        self.blur.toggled.connect(lambda: self.menu.setCurrentWidget(self.blurMenu))
        self.gray.toggled.connect(lambda: self.menu.setCurrentWidget(self.GrayMenu))
        self.edge.toggled.connect(lambda: self.menu.setCurrentWidget(self.EdgeMenu))
        
        self.proses.clicked.connect(self.process_image)
        
        self.inputBiner1.valueChanged.connect(self.inputBiner2.setValue)
        self.inputBiner2.valueChanged.connect(self.inputBiner1.setValue)
        
        self.inputBrigth1.valueChanged.connect(self.inputBrigth2.setValue)
        self.inputBrigth2.valueChanged.connect(self.inputBrigth1.setValue)
        
        self.horizontalSlider.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.horizontalSlider.setValue)

    def openImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", 
                                                options=options)
        if fileName:
            self.original_image = cv2.imread(fileName)
            self.displayImage(self.original_image, self.gambar1)
            self.displayHistogram(self.original_image, self.histogram1)

    def saveImage(self):
        if self.processed_image is not None:
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", 
                                                    "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", 
                                                    options=options)
            if fileName:
                cv2.imwrite(fileName, self.processed_image)

    def displayImage(self, img, label):
        if img is None:
            return
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        label.setPixmap(QPixmap.fromImage(outImage))
        label.setScaledContents(True)

    def displayHistogram(self, img, label):
        color = ('b', 'g', 'r')
        fig = plt.figure()
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_rgb()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        label.setPixmap(QtGui.QPixmap.fromImage(qImg))
        label.setScaledContents(True)
        
        plt.close(fig)

    def process_image(self):
        try:
            if self.biner.isChecked():
                threshold = self.inputBiner2.value()
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                _, processed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                
            elif self.negatif.isChecked():
                processed = cv2.bitwise_not(self.original_image)
                
            elif self.Brigth.isChecked():
                brightness = self.inputBrigth2.value()
                processed = cv2.add(self.original_image, np.array([brightness, brightness, brightness]))
                
            elif self.Flip.isChecked():
                if self.vertikalRadio.isChecked():
                    processed = cv2.flip(self.original_image, 0)
                elif self.horizontalRadio.isChecked():
                    processed = cv2.flip(self.original_image, 1)
                
            elif self.Sharp.isChecked():
                if self.radioButton_2.isChecked():  # Basic Sharpening
                    kernel = np.array([[ 0, -1,  0],
                                        [-1,  5, -1],
                                        [ 0, -1,  0]])
                elif self.radioButton_3.isChecked():  # Enhanced Sharpening
                    kernel = np.array([[-1, -1, -1],
                                        [-1,  9, -1],
                                        [-1, -1, -1]])
                
                processed = cv2.filter2D(self.original_image, -1, kernel)
                
            elif self.blur.isChecked():
                kernel_size = self.spinBox.value()
                if kernel_size % 2 == 0:
                    kernel_size += 1
                processed = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
            # Grayscale
            elif self.gray.isChecked():
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Edge Detection
            elif self.edge.isChecked():
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(self.original_image,(3,3), sigmaX=0, sigmaY=0)
                
                if self.sobel.isChecked():
                    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    processed = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

                elif self.Prewitt.isChecked():
                    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                    img_prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
                    img_prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
                    processed_x = cv2.convertScaleAbs(img_prewittx)
                    processed_y = cv2.convertScaleAbs(img_prewitty)
                    processed = cv2.addWeighted(processed_x, 0.5, processed_y, 0.5, 0)

                elif self.Roberts.isChecked():
                    kernelx = np.array([[1, 0], [0, -1]])
                    kernely = np.array([[0, 1], [-1, 0]])
                    image=gray/255.0
                    vertical = ndimage.convolve(image, kernelx)
                    horizontal = ndimage.convolve(image, kernely)
                    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical)) 
                    edged_img*=255
                    processed = cv2.convertScaleAbs(edged_img)

                elif self.Canny.isChecked():
                    processed = cv2.Canny(img_blur, 100, 200)
                else:
                    return

            self.processed_image = processed
            self.displayImage(processed, self.gambar2)
            self.displayHistogram(processed, self.histogram2)

            
        except Exception as e:
            print(f"Error :  {str(e)}")

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.timer.start(30)
            else:
                QMessageBox.warning(self, "Error", "Gagal Buka Kamera")
                self.camera = None

    def stop_camera(self):
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.gambar1.clear()
            self.histogram1.clear()

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.original_image = frame
            self.displayImage(frame, self.gambar1)
            self.displayHistogram(frame, self.histogram1)
            if any([self.biner.isChecked(), self.negatif.isChecked(), 
                   self.Brigth.isChecked(), self.Flip.isChecked(),
                   self.Sharp.isChecked(), self.blur.isChecked(),
                   self.gray.isChecked(), self.edge.isChecked()]):
                self.process_image()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
