import numpy as np
from matplotlib.pyplot import axvline, axhline
import matplotlib.pyplot as plt
from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
import matplotlib.backends.backend_qt4agg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar)
from PyQt4.QtGui import *
import serial  # import Serial Library
#from drawnow import *
import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import RealTime
import poweroff
import threading
from bluepy import btle
from PyQt4.QtCore import QObject,pyqtSignal

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

class Main(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)
        self.Real = RealTime.RealTime()
        self.Power=poweroff.poweroff()    
        #self.Real.set_GP_instance(self)
        
        #self.textBrowser.setText( "stdouterr" )
        #self.textBrowser.insertPlainText("yA Rab \n")
        
        XStream.stdout().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stdout().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        XStream.stderr().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stderr().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        
        #self.emgplot = pg.PlotWidget( name='EMGplot' )
        self.emgplot.setRange( QtCore.QRectF( -50, -200, 1000, 1400 ) )
        self.emgplot.disableAutoRange()
        self.emgplot.setTitle( "EMG" )

        self.refreshRate = 0.05
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
           
       

        self.lastUpdateTime = time.time()
        #self.show()
        
        self.pushButton.clicked.connect(self.Real.start_MYO)        
        self.pushButton_2.clicked.connect( self.start_thread1)
        self.pushButton_3.clicked.connect( self.stop_thread1)
        self.pushButton_4.clicked.connect( self.disconnect_MYO)
        self.pushButton_5.clicked.connect(self.Power.power_off)
        self.pushButton_6.clicked.connect( self.clear_textBrowser )
        self.pushButton_7.clicked.connect( self.start_thread2 )
        self.pushButton_8.clicked.connect( self.stop_thread2 )
        self.pushButton_9.clicked.connect(self.file_save_csv)
        self.pushButton_10.clicked.connect(self.browse_pickle)
        self.pushButton_11.clicked.connect(self.start_thread3)
        self.pushButton_12.clicked.connect(self.stop_thread3)
        #self.pushButton_4.setStyleSheet("background-color: red")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
        self.label.setPalette(palette)

       
        self.thread1 = None
        self.thread2 = None
        self.event_stop_thread1 = threading.Event()
        self.event_stop_thread2 = threading.Event()
        self.event_stop_thread3 = threading.Event()
        
    def start_thread1(self):
        self.Real.Flag_Predict= True
        self.Real.b = np.empty( [0, 8] )
        self.Real.predictions_array=[]
        self.event_stop_thread1.clear()
        self.thread1 = threading.Thread(target = self.loop1)
        self.thread1.start()
        
    def start_thread2(self):
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Graph = True
        self.event_stop_thread2.clear()
        self.thread2 = threading.Thread(target = self.loop2)
        self.thread2.start()
    def start_thread3(self):
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Graph0 = True
        self.event_stop_thread3.clear()
        self.thread3 = threading.Thread(target = self.loop3)
        self.thread3.start()
        
    def loop1(self):
        while not self.event_stop_thread1.is_set():
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                #continue
                print(self.Real.predictions_array)
                #print (self.Real.p)
                #print (self.Real.prediction)
                
        
    def loop2(self):
        while not self.event_stop_thread2.is_set():
            self.update_plots()
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue
    def loop3(self):
        while not self.event_stop_thread3.is_set():
            self.updater()
            if self.Real.myo_device.services.waitForNotifications( 1 ):
                continue
    def loop4(self):
        
        while  self.Real.b.shape[0] < self.records:
            print (self.Real.b.shape[0])
            if self.Real.myo_device.services.waitForNotifications(1):
                continue
            
        np.savetxt(str(self.path)+".csv", self.Real.b, delimiter="," ,fmt='%10.5f')
        self.Real.b= np.empty([0,8])
        print ("saved Sucessfully at %s" % self.path)
        #self.stop_thread4()
                
  
    def stop_thread1(self):
        self.event_stop_thread1.set()
        self.thread1.join()
        self.thread1 = None
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Predict =False
    def stop_thread2(self):
        self.event_stop_thread2.set()
        self.thread2.join()
        self.thread2 = None
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Graph= False
    def stop_thread3(self):
        self.event_stop_thread3.set()
        self.thread3.join()
        self.thread3 = None
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Graph0= False
    def stop_thread4(self):
        self.event_stop_thread4.set()
        self.thread4.join()
        self.thread4 = None

        
    def clear_textBrowser(self):          
        self.textBrowser.clear()
        
    def disconnect_MYO(self):
        print ("attempting to Disconnect")
        self.Real.myo_device.services.vibrate( 1 )  # short vibration
        #btle.Peripheral.disconnect()
        self.Real.myo_device.services.disconnect_MYO()
        print ("Successfully Disconnected")


    def updater(self):
     
       
        for i in range( 8 ):
            self.emgcurve0[i].plot(pen=(i, 10)).setData( self.Real.b[:,i] )
      
        #self.EMG1.plot(self.Real.b[:,0], pen=pen1,clear=True)
        #self.EMG2.plot(self.Real.b[:,1], pen=pen2, clear=True)
        #app.processEvents()
        if self.Real.b.shape[0] % 50 ==0 :
            self.Real.b = np.delete(self.Real.b,slice(0,20), axis=0)
   
        

    def update_plots(self):
        #ctime = time.time()
        #if (ctime - self.lastUpdateTime) >= self.refreshRate:
        for i in range( 8 ):
            self.emgcurve[i].setData( self.Real.b[:,i] )
            #for i in range( 4 ):
                #self.oricurve[i].setData( self.listener.orientation.data[i, :] )
            #for i in range( 3 ):
                #self.acccurve[i].setData( self.listener.acc.data[i, :] )
            #self.lastUpdateTime = ctime
            #self.Real.b = np.empty( [0, 8] )
            app.processEvents()
            
        if self.Real.b.shape[0] % 5 ==0 :
            self.Real.b = np.delete(self.Real.b,[0], axis=0)




    def browse_pickle(self):
        self.flag1=1

        filepath = QtGui.QFileDialog.getOpenFileName(self, 'Single File', "",'*.pickle')
        f= str(filepath)
        if f != "":
            spf = wave.open(f, 'r')
        import contextlib

        with contextlib.closing(wave.open(f, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            print "Duration is " , duration



    def file_save_csv(self):
      

        self.path = QtGui.QFileDialog.getSaveFileName(self, 'Save Point', "", '*.csv')
        print (" Path = %s" %self.path)
        self.records=int(self.lineEdit.text())
        self.Real.b= np.empty([0,8])
        self.event_stop_thread4 = threading.Event()
        self.event_stop_thread4.clear()
        self.thread4 = threading.Thread(target = self.loop4)
        self.thread4.start()
        

        #file.close()



if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())



