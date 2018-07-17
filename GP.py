import numpy as np
import matplotlib.pyplot as plt
from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QObject,pyqtSignal
from PyQt4.QtGui import *
import serial  # import Serial Library
#from drawnow import *
import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import EMG
import poweroff
import threading
from bluepy import btle
import CV
import EMG_Model
import CV_realtime
#import collections
import Queue as queue ##If python 2
#import queue  ##If python 3
import pandas as pd
import cv2
Ui_MainWindow, QMainWindow = loadUiType('GP.ui')

class XStream(QObject):
    _stdout = None
    _stderr = None

    messageWritten = pyqtSignal(str)

    def flush( self ):
        pass

    def fileno( self ):
        return -1

    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(unicode(msg))

    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr
class OwnImageWidget( QtGui.QWidget ):
    def __init__(self, parent=None):
        super( OwnImageWidget, self ).__init__( parent )
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize( sz )
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin( self )
        if self.image:
            qp.drawImage( QtCore.QPoint( 0, 0 ), self.image )
        qp.end()

class LoadImageThread( QtCore.QThread ):
    def __init__(self, file, w, h):
        QtCore.QThread.__init__( self )
        self.file = file
        self.w = w
        self.h = h
    def __del__(self):
        self.wait()
    def run(self):
        self.emit( QtCore.SIGNAL( 'showImage(QString, int, int)' ), self.file, self.w, self.h )
class LoadImageThread2( QtCore.QThread ):
    def __init__(self, file, w, h):
        QtCore.QThread.__init__( self )
        self.file = file
        self.w = w
        self.h = h
    def __del__(self):
        self.wait()
    def run(self):
        self.emit( QtCore.SIGNAL( 'showImage2(QString, int, int)' ), self.file, self.w, self.h )
class Main(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)
        self.Real = EMG.RealTime()
        self.Power=poweroff.poweroff()
        self.EMG_Modeling = EMG_Model.EMG_Model()
        self.cv = CV.CV()
        #self.Real.set_GP_instance(self)
        
        ##TextBrowser
        XStream.stdout().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stdout().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        XStream.stderr().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stderr().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        
        #self.emgplot = pg.PlotWidget( name='EMGplot' )
        self.emgplot.setRange( QtCore.QRectF( -50, -200, 1000, 1400 ) )
        self.emgplot.disableAutoRange()
        self.emgplot.setTitle( "EMG" )

        self.emgcurve = []
        for i in range( 8 ):
            c = self.emgplot.plot( pen=(i, 10) )
            c.setPos( 0, i * 150 )
            self.emgcurve.append( c )
        
        self.emgcurve0 = [self.EMG1,self.EMG2,self.EMG3,self.EMG4,self.EMG5\
                           ,self.EMG6,self.EMG7,self.EMG8]
        for i in range (8):
            self.emgcurve0[i].plotItem.showGrid(True, True, 0.7)
            #self.emgcurve0[i].plotItem.setRange(yRange=[0, 1])


        self.pushButton.clicked.connect(self.Real.start_MYO)
        self.pushButton_2.clicked.connect( self.start_thread2 )  # Start Predict
        self.pushButton_3.clicked.connect( self.stop_thread2 )  # Stop Predict
        self.pushButton_4.clicked.connect( self.disconnect_MYO)
        self.pushButton_5.clicked.connect(self.Power.power_off)
        self.pushButton_6.clicked.connect( self.clear_textBrowser )
        self.pushButton_7.clicked.connect( self.start_thread1 )# start Graph1
        self.pushButton_8.clicked.connect( self.stop_thread1 )
        self.pushButton_9.clicked.connect( self.file_save_csv )
        self.pushButton_11.clicked.connect( self.start_thread0 )
        self.pushButton_12.clicked.connect( self.stop_thread0 )
        self.pushButton_10.clicked.connect( self.saveEMGModel )
        self.pushButton_10.setStyleSheet( "background-color: red" )
        self.pushButton_13.clicked.connect( self.browseCSVEMGModel1 )
        self.pushButton_14.clicked.connect( self.browseCSVEMGModel2 )
        self.pushButton_15.clicked.connect( self.browseCSVEMGModel3 )
        self.pushButton_16.clicked.connect( self.browseCSVEMGModel4 )
        self.pushButton_21.clicked.connect( self.joinCSV1 )
        self.pushButton_22.clicked.connect( self.joinCSV2 )
        self.pushButton_23.clicked.connect( self.saveJoinCSV )
        self.pushButton_17.clicked.connect( self.browsePickleEMGModel1 )
        self.pushButton_18.clicked.connect( self.browsePickleEMGModel2 )
        self.pushButton_19.clicked.connect( self.browseCVModel )
        self.pushButton_20.clicked.connect( self.start_thread4 )
        self.pushButton_20.setStyleSheet( "background-color: green" )
        self.pushButton_24.clicked.connect( self.stop_thread4 )
        self.pushButton_24.setStyleSheet( "background-color: red" )
        self.pushButton_25.clicked.connect( QtCore.QCoreApplication.instance().quit )
        self.path1 = self.path2 = self.path3 = self.path4 = self.path5 = self.path6 = self.path7 = self.path8 = None

        ## To change text Color to Red Color
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
        self.label.setPalette(palette)

        ##############################################################
        self.CV_realtime = CV_realtime.MyThread()
        self.CV_realtimeFlag = None
        self.CV_realtimeFlag2 = 0
        self.capture_thread = None
        self.q = queue.Queue()
        ##############################################################
        self.startButton.clicked.connect( self.start_camera )
        self.pushButton_26.clicked.connect( self.close_camera )
        self.pushButton_27.clicked.connect( self.start_cvRealtime )
        self.pushButton_28.clicked.connect( self.stop_cvRealtime )
        self.pushButton_28.setEnabled( False )
        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget( self.ImgWidget )
        self.timer = QtCore.QTimer( self )
        self.timer.timeout.connect( self.update_frame )
        self.timer.start( 1 )
        ####
        self.pushButton_29.clicked.connect( self.browsePickleEMGModel3 )
        self.pushButton_30.clicked.connect( self.start_thread5 )
        self.pushButton_31.clicked.connect( self.stop_thread5 )
        self.pushButton_30.setStyleSheet( "background-color: green" )
        self.pushButton_31.setStyleSheet( "background-color: red" )
        ###################################################################################################################
        self.thread1 = None
        self.thread2 = None
        self.event_stop_thread0 = threading.Event()
        self.event_stop_thread1 = threading.Event()
        self.event_stop_thread2 = threading.Event()
        self.event_stop_thread3 = threading.Event()
        self.event_stop_thread4 = threading.Event()
        self.event_stop_thread5 = threading.Event()

    def start_camera(self):
        self.ImgWidget.setHidden( False )
        self.running = True
        self.capture_thread = threading.Thread( target=self.grab, args=(0, self.q, 1920, 1080, 30) )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.startButton.setEnabled( False )
        self.pushButton_26.setEnabled( True )
        self.startButton.setText( 'Starting...' )
        self.startButton.setText( 'Camera is live' )

    def grab(self, cam, queue, width, height, fps):
        self.capture = cv2.VideoCapture( cam )
        self.capture.set( cv2.CAP_PROP_FRAME_WIDTH, width )
        self.capture.set( cv2.CAP_PROP_FRAME_HEIGHT, height )
        self.capture.set( cv2.CAP_PROP_FPS, fps )

        while (self.running):
            frame = {}
            # Get the original frame from video capture
            retval, original_frame = self.capture.read()
            # Resize the frame to fit the imageNet default input size
            if self.CV_realtimeFlag is not None :
                self.CV_realtime.frame_to_predict = cv2.resize( original_frame, (224, 224) )
                if self.checkBox.isChecked():
                    # Add text label and network score to the video captue
                    cv2.putText( original_frame, "Label: %s | Score: %.2f" % (self.CV_realtime.label, self.CV_realtime.score), (15, 60), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 2 )

                cv2.putText(original_frame, "Name: %s| Class: %d " % (self.CV_realtime.grasp_name, self.CV_realtime.grasp_number),(0, 25),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2 )
            self.capture.grab()
            # retval, img = capture.retrieve( 0 )
            frame["img"] = original_frame

            if queue.qsize() < 10:
                queue.put( frame )
            else:
                print
                queue.qsize()

    def start_cvRealtime(self):
        # self.CV_realtime = CV_realtime.MyThread()
        self.CV_realtimeFlag = 1
        if (self.CV_realtimeFlag2 == 0):
            self.CV_realtimeFlag2 = 1
            self.CV_realtime.daemon = True
            self.CV_realtime.start()
       
        self.pushButton_27.setEnabled( False )
        self.pushButton_28.setEnabled( True )

    def stop_cvRealtime(self):
        self.CV_realtime.classficication = False
        self.pushButton_27.setEnabled( True )
        self.pushButton_28.setEnabled( False )
        self.CV_realtime.frame_to_predict = None
        self.CV_realtimeFlag = None
        # self.CV_realtime = CV_realtime.MyThread()
        # self.capture.release()
        # cv2.destroyAllWindows()

    def update_frame(self):
        if not self.q.empty():
            # self.startButton.setText( 'Camera is live' )
            frame = self.q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float( self.window_width ) / float( img_width )
            scale_h = float( self.window_height ) / float( img_height )
            scale = min( [scale_w, scale_h] )

            if scale == 0:
                scale = 1

            img = cv2.resize( img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC )
            img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage( img.data, width, height, bpl, QtGui.QImage.Format_RGB888 )
            self.ImgWidget.setImage( image )

    def close_camera(self, event):
        # global running
        self.ImgWidget.setHidden( True )
        self.running = False
        self.startButton.setText( 'Start Video' )
        self.startButton.setEnabled( True )
        self.pushButton_26.setEnabled( False )
        self.CV_realtime.frame_to_predict = None
        # cv2.destroyAllWindows()
    def ReadEMG(self):
        while (True):
            #time.sleep(0.05)
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue
            break
    def start_thread0(self):## Graph0
        self.Real.EMG = np.empty( [0, 8] )
        self.event_stop_thread0.clear()
        #threading.Thread( target=lambda: self.ReadEMG() ).start()
        #self.flag_thread0 = True
        self.thread0 = threading.Thread(target = self.loop0)
        self.thread0.daemon = True
        self.thread0.start()
        
    def start_thread1(self):##Graph1
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        #self.flag_thread1 = True
        self.event_stop_thread1.clear()
        self.thread1 = threading.Thread(target = self.loop1)
        self.thread1.daemon = True
        self.thread1.start()
    def start_thread2(self):##Predict
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        self.cv.q.queue.clear()
        #threading.Thread( target=self.ReadEMG() ).start()
        #self.flag_thread2 = True
        self.event_stop_thread2.clear()
        self.thread2 = threading.Thread(target = self.loop2)
        self.thread2.start()

    def start_thread4(self):  ##System
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        self.cv.q.queue.clear()
        #self.flag_thread4 = True
        self.event_stop_thread4.clear()
        self.thread4 = threading.Thread( target=self.loop4 )
        self.thread4.start()
    def start_thread5(self):  ## Online_System
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        self.CV_realtime.q.queue.clear()
        self.CV_realtime.stage = 0
        self.CV_realtime.corrections = 0
        self.CV_realtime.grasp1 = None
        self.CV_realtimeFlag = 1
        if (self.CV_realtimeFlag2 == 0):
            self.CV_realtimeFlag2 = 1
            self.CV_realtime.daemon = True
            self.CV_realtime.start()
        elif (self.CV_realtimeFlag2 == 1):
            pass
        #self.flag_thread4 = True
        self.event_stop_thread5.clear()
        self.thread5 = threading.Thread( target=self.loop5 )
        self.thread5.start()

    def loop0(self):
        while not self.event_stop_thread0.is_set():
            time.sleep( 0.5)
            self.update_Graph0()
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue
    
    def loop1(self):#Graph1
        while not self.event_stop_thread1.is_set():
            self.update_Graph1()
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue
            time.sleep( 0.5 )
 

    def loop2(self):## Predict
        while not self.event_stop_thread2.is_set():
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                #continue
                c = self.Real.predict( path=self.path7 )
                if  c.size ==1:
                    self.cv.q.put( int( c ) )
                    print (self.cv.q.queue)
                    self.someFunctionCalledFromAnotherThread2( int( c ) )
            #time.sleep( 0.01 )

    def loop4(self):  ##System
        while not self.event_stop_thread4.is_set():
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                c = self.Real.predict( path=self.path8 )
                if  c.size ==1 :
                    self.cv.q.put( int( c ) )
                    print (self.cv.q.queue)
                    self.cv.Main_algorithm( path1=self.path9 )
            #time.sleep( 0.01 )

    def loop5(self):  ##Online_System
        while not self.event_stop_thread5.is_set():
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                c = self.Real.predict( path=self.path10 )
                if  c.size == 1:
                    self.CV_realtime.q.put( int( c ) )
                    print( self.cv.q.queue )
                    self.CV_realtime.Main_algorithm()
                    if self.CV_realtime.final is not None:
                        self.someFunctionCalledFromAnotherThread( self.CV_realtime.final )
                    # time.sleep( 0.01 )
    def someFunctionCalledFromAnotherThread(self,grasp):
        if grasp == 1:
            thread = LoadImageThread( file="screenshots/pinch.png", w=204, h=165 )
            self.connect( thread, QtCore.SIGNAL( "showImage(QString, int, int)" ), self.showImage )
            thread.start()
        elif grasp == 2:
            thread = LoadImageThread( file="screenshots/palmar_neutral.png", w=238, h=158 )
            self.connect( thread, QtCore.SIGNAL( "showImage(QString, int, int)" ), self.showImage )
            thread.start()
        elif grasp == 3:
            thread = LoadImageThread( file="screenshots/tripod.png", w=242, h=162 )
            self.connect( thread, QtCore.SIGNAL( "showImage(QString, int, int)" ), self.showImage )
            thread.start()
        elif grasp == 4:
            thread = LoadImageThread( file="screenshots/palmar_pronated.png", w=219, h=165 )
            self.connect( thread, QtCore.SIGNAL( "showImage(QString, int, int)" ), self.showImage)
            thread.start()

    def showImage(self, filename, w, h):
        pixmap = QtGui.QPixmap( filename ).scaled( w, h )
        self.label_13.setPixmap( pixmap )
        self.label_13.repaint()
    def someFunctionCalledFromAnotherThread2(self,EMG_class):
        if EMG_class == 1:
            thread = LoadImageThread2( file="screenshots/finger_spread.png", w=278, h=299 )
            self.connect( thread, QtCore.SIGNAL( "showImage2(QString, int, int)" ), self.showImage2 )
            thread.start()
        elif EMG_class == 2:
            thread = LoadImageThread2( file="screenshots/wrist_extension.png", w=348, h=302 )
            self.connect( thread, QtCore.SIGNAL( "showImage2(QString, int, int)" ), self.showImage2 )
            thread.start()
        elif EMG_class == 3:
            thread = LoadImageThread2( file="screenshots/wrist_ulnar_deviation.png", w=283, h=254)
            self.connect( thread, QtCore.SIGNAL( "showImage2(QString, int, int)" ), self.showImage2 )
            thread.start()
        elif EMG_class == 0:
            thread = LoadImageThread2( file="screenshots/rest.png", w=353, h=254 )
            self.connect( thread, QtCore.SIGNAL( "showImage2(QString, int, int)" ), self.showImage2)
            thread.start()

    def showImage2(self, filename, w, h):
        pixmap = QtGui.QPixmap( filename ).scaled( w, h )
        self.label_15.setPixmap( pixmap )
        self.label_15.repaint()

    def stop_thread0(self):
        self.event_stop_thread0.set()
        self.thread0.join()
        self.thread0 = None
        self.Real.EMG = np.empty( [0, 8] )

    def stop_thread1(self):
        self.event_stop_thread1.set()
        self.thread1.join()
        self.thread1 = None
        self.Real.EMG = np.empty( [0, 8] )

  
    def stop_thread2(self):
        self.event_stop_thread2.set()
        self.thread2.join()
        self.thread2 = None
        self.Real.EMG = np.empty( [0, 8] )
        #self.Real.Flag_Graph= False
    def stop_thread3(self):
        self.event_stop_thread3.set()
        self.thread3.join()
        self.thread3 = None
        self.Real.EMG = np.empty( [0, 8] )
        #self.Real.Flag_Graph0= False
    def stop_thread4(self): ## System
        self.event_stop_thread4.set()
        self.thread4.join()
        self.thread4 = None
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        self.cv.q.queue.clear()
        self.c = np.array( [] )
        print( ("Thread Of System Closed ") )

    def stop_thread5(self):  ##Online_System
        self.event_stop_thread5.set()
        self.thread5.join()
        self.thread5 = None
        self.Real.emg_total = np.empty( [0, 8] )
        self.Real.EMG = np.empty( [0, 8] )
        self.CV_realtime.q.queue.clear()
        print( ("Thread Of Online System is Closed ") )
        
    def clear_textBrowser(self):          
        self.textBrowser.clear()
        
    def disconnect_MYO(self):
        print ("attempting to Disconnect")
        self.Real.myo_device.services.vibrate( 1 )  # short vibration
        #btle.Peripheral.disconnect()
        self.Real.myo_device.services.disconnect_MYO()
        print ("Successfully Disconnected")


    def update_Graph0(self):

        for i in range( 8 ):
            self.emgcurve0[i].plot(pen=(i, 10)).setData( self.Real.EMG[:,i] )
      
        #self.EMG1.plot(self.Real.b[:,0], pen=pen1,clear=True)
        #self.EMG2.plot(self.Real.b[:,1], pen=pen2, clear=True)
        #app.processEvents()
        if self.Real.EMG.shape[0] >=150 :
            self.Real.EMG = np.delete(self.Real.EMG,slice(0,20), axis=0)
   
        

    def update_Graph1(self):
        
        for i in range( 8 ):
            self.emgcurve[i].setData( self.Real.EMG[:,i] )
            app.processEvents()
            
        if self.Real.EMG.shape[0] % 5 ==0 :
            self.Real.EMG = np.delete(self.Real.EMG,[0], axis=0)

    def file_save_csv(self):

        self.path = QtGui.QFileDialog.getSaveFileName( self, 'Save Point', "", '*.csv' )
        print (" Path = %s" % self.path)
        self.records = int( self.lineEdit.text() )
        self.Real.EMG = np.empty( [0, 8] )
        self.Real.emg_total = np.empty( [0, 8] )
        self.event_stop_thread3 = threading.Event()
        self.event_stop_thread3.clear()
        self.thread3 = threading.Thread( target=self.save_loop)
        self.thread3.start()

    def save_loop(self):
        while self.Real.EMG.shape[0] < self.records:
            print (self.Real.EMG.shape[0])
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue

        np.savetxt( str( self.path ) + ".csv", self.Real.EMG, delimiter=",", fmt='%10.5f' )
        self.Real.EMG = np.empty( [0, 8] )
        print ("saved Sucessfully at %s" % self.path)
        self.thread3 = None

    def browseCSVEMGModel1(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_2.setText( filepath )
        self.path1 = str( filepath )
        print (" Path = %s" % self.path1)
        # self.records = int( self.lineEdit.text() )

    def browseCSVEMGModel2(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_6.setText( filepath )
        self.path2 = str( filepath )
        print (" Path = %s" % self.path2)

    def browseCSVEMGModel3(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_7.setText( filepath )
        self.path3 = str( filepath )
        print (" Path = %s" % self.path3)

    def browseCSVEMGModel4(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_8.setText( filepath )
        self.path4 = str( filepath )
        print (" Path = %s" % self.path4)

    def saveEMGModel(self):
        if not self.path1 == None and not self.path2 == None and not self.path3 == None and not self.path4 == None:
            filepath = QtGui.QFileDialog.getSaveFileName( self, 'Save Point', "", '*.pickle' )
            filepath = filepath  +".pickle"
            print ((" path is  = %s" % str(filepath)))
            self.EMG_Modeling.all_steps( path1=self.path1, path2=self.path2, path3=self.path3, path4=self.path4,
                                         file_name=str( filepath ) )
            print (" Saved SuccessFully at = %s" % filepath)

    def joinCSV1(self):

        self.path5 = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_9.setText( self.path5 )
        print (" Path = %s" % self.path5)

    def joinCSV2(self):

        self.path6 = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_10.setText( self.path6 )
        print (" Path = %s" % self.path6)

    def saveJoinCSV(self):
        if not self.path5 == None and not self.path6 == None:
            filepath = QtGui.QFileDialog.getSaveFileName( self, 'Save Point', "", '*.csv' )

            a = pd.read_csv( str( self.path5 ), header=None, index_col=False )
            b = pd.read_csv( str( self.path6 ), header=None, index_col=False )
            c = pd.concat( [a, b] )
            c.to_csv( str( filepath ) + ".csv", index=False, header=None )
            print (" Saved SuccessFully at = %s" % filepath)

    def browsePickleEMGModel1(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.pickle' )
        self.lineEdit_3.setText( filepath )
        self.path7 = str( filepath )
        print (" Path = %s" % self.path7)

    def browsePickleEMGModel2(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.pickle' )
        self.lineEdit_4.setText( filepath )
        self.path8 = str( filepath )
        print (" Path = %s" % self.path8)

    def browseCVModel(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.h5' )
        self.lineEdit_5.setText( filepath )
        self.path9 = str( filepath )
        print (" Path = %s" % self.path9)
    def browsePickleEMGModel3(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.pickle')
        self.lineEdit_11.setText( filepath)
        self.path10 = str( filepath )
        print((" Path = %s" % self.path10))




if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.setWindowIcon( QtGui.QIcon( 'screenshots/x.png' ) )
    main.show()
    sys.exit(app.exec_())



