
# for memory error >> try to change the float 64 to float 32 
import numpy as np
#import scipy.io as sio
import pandas as pd
#import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,filtfilt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from scipy import stats
#from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import sys
#import test
import time
#from test import MyoRaw
import open_myo as myo
import threading
#import GP
class RealTime():

    def __init__(self):
        #super(RealTime, self).__init__()
        #self.setupUi(self)

        self.EMG = np.empty( [0, 8] )
        self.predictions_array = []
        self.p=np.empty([0,8])
        self.emg_total = np.empty( [0, 8] )
        self.iteration = 0
        self.Flag_Graph0=None
        self.Flag_Graph=None
        self.Flag_Predict =None
        self.prediction = None
        self.stop_request =True
        
        #self.set_GP_instance(GP)
        
    def set_GP_instance(self,GP):
        self.GP=GP
        

#search on Hampel filter to remove spikes. and make notch filter on 50 hz
    def filteration (self,data,sample_rate=2000.0,cut_off=20.0,order=5,ftype='highpass'):
        nyq = .5 * sample_rate
        b,a= butter(order,cut_off/nyq,btype=ftype)
        d= lfilter(b,a,data,axis=0)
        return pd.DataFrame(d)

    def MES_analysis_window(self,df, width, tau, win_num):
        df_2 = pd.DataFrame()
        start = win_num * tau
        end = start + width
        df_2 = df.iloc[start:end]
        return end, df_2

    def features_extraction(self,df, th=0):
        # F1 : mean absolute value (MAV)
        MAV = abs( df.mean( axis=0 ) )

        MAV = list( MAV )
        WL = []
        SSC = []
        ZC = []
        for col, series in df.iteritems():
            # F2 : wave length (WL)
            s = abs( np.array( series.iloc[:-1] ) - np.array( series.iloc[1:] ) )
            WL_result = np.sum( s )
            WL.append( WL_result )

            # F3 : zero crossing(ZC)
            _1starray = np.array( series.iloc[:-1] )
            _2ndarray = np.array( series.iloc[1:] )
            ZC.append( ((_1starray * _2ndarray < 0) & (abs( _1starray - _2ndarray ) >= th)).sum() )

            # F4 : slope sign change(SSC)
            _1st = np.array( series.iloc[:-2] )
            _2nd = np.array( series.iloc[1:-1] )
            _3rd = np.array( series.iloc[2:] )
            SSC.append( ((((_2nd - _1st) * (_2nd - _3rd)) > 0) & (
            ((abs( _2nd - _1st )) >= th) | ((abs( _2nd - _3rd )) >= th))).sum() )

        features_array = np.array( [MAV, WL, ZC, SSC] ).T
        return features_array




    def get_predictors(self,emg,width=512,tau=128):

        x=[];
        end=0; win_num=0;
        while((len(emg)-end) >= width):
            end,window_df=self.MES_analysis_window(emg,width,tau,win_num)
            win_num=win_num + 1

            ff=self.features_extraction(window_df)
            x.append(ff)

        predictors_array=np.array(x)

        nsamples, nx, ny = predictors_array.shape
        predictors_array_2d = predictors_array.reshape((nsamples,nx*ny))

        return np.nan_to_num(predictors_array_2d)



    """
    def predict(self,emg,tau=128):
        #emg = np.random.rand(512,8)
        #global b,emg_total,iteration
        self.emg_total= np.append(self.emg_total,self.EMG,axis=0)
        print (self.emg_total.shape)
        if self.emg_total.shape[0] == 512:
            data= pd.DataFrame(self.emg_total)
            filtered_emg=self.filteration (data,sample_rate=200)
            predictors_test = self.get_predictors(filtered_emg)
            self.emg_total = self.emg_total[128:]
            filename = 'EMG_hanna_model2.pickle'
            pickled_clf=joblib.load(filename)
            self.EMG= np.empty([0,8]) 
            return pickled_clf.predict(predictors_test)
        self.EMG= np.empty([0,8]) 
        return 0
    """
    def predict(self , path):
        if self.emg_total.shape[0] >= 512:
          self.flag_Predict =1
          #print ("Hiiii")
          self.emg_total = np.append( self.emg_total, self.EMG[:128], axis=0 )
          self.EMG = self.EMG[128:]
          data = pd.DataFrame( self.emg_total )
          filtered_emg = self.filteration( data, sample_rate=200 )
          predictors_test = self.get_predictors( filtered_emg )
          self.emg_total = self.emg_total[128:]
          filename = path
          pickled_clf = joblib.load( filename )
          return pickled_clf.predict( predictors_test )
        else :
          n= self.EMG.shape[0]
          self.emg_total = np.append( self.emg_total, self.EMG[:n], axis=0 )
          self.EMG = self.EMG[n:]
          return np.array( [] )

    

    def start_MYO(self):
        myo_mac_addr = myo.get_myo()
        print("MAC address: %s" % myo_mac_addr)
     
        self.myo_device = myo.Device()
        self.myo_device.services.sleep_mode( 1 )  # never sleep
        self.myo_device.services.set_leds( [128, 128, 255], [128, 128, 255] )  # purple logo and bar LEDs)
        self.myo_device.services.vibrate( 1 )  # short vibration
        fw = self.myo_device.services.firmware()
        print("Firmware version: %d.%d.%d.%d   \n" % (fw[0], fw[1], fw[2], fw[3]))
     
        batt = self.myo_device.services.battery()
        print("Battery level: %d" % batt)
       
        # myo_device.services.emg_filt_notifications()
        self.myo_device.services.emg_raw_notifications()
        # myo_device.services.imu_notifications()
        # myo_device.services.classifier_notifications()
        # myo_device.services.battery_notifications()
        self.myo_device.services.set_mode( myo.EmgMode.RAW, myo.ImuMode.OFF, myo.ClassifierMode.OFF )
        self.myo_device.add_emg_event_handler( self.process_emg )
        # myo_device.add_emg_event_handler(led_emg)
        # myo_device.add_imu_event_handler(process_imu)
        # myo_device.add_sync_event_handler(process_sync)
        # myo_device.add_classifier_event_hanlder(process_classifier)
       




    def final(self,emg):
        print (":D")

        print (emg.shape)
        # print emg[:,0] ## if you want a single channel
        #global b
        self.EMG = np.empty( [0, 8] )

    def process_emg(self,emg):
        # unfortunately the Filtered Array provide 1 array of 8 element at a time  ==> in te Form of Tuple
        # while The RAW_EMG provide 2 array at a time 8 elements each , ===> in the form of list that contains 2 tuples
        # print(emg)
        #global b
        ## for RAW_EMG
        self.EMG = np.append( self.EMG, emg, axis=0 )
        #print (self.EMG.shape[0])
        #if self.Flag_Predict == True and self.EMG.shape[0] == 128 :
            #self.predictions_array.append(self.predict( self.EMG ))
            #self.p = np.append(self.p,self.predict(self.EMG), axis=0)
            #c=self.predict( self.EMG )
            
            #self.prediction= self.predict( self.EMG )
            # final(b)
            
        #elif self.Flag_Graph == True and self.EMG.shape[0] ==1000 :
            #self.EMG= np.empty([0,8])
            
            

            ## For Filtered_EMG
            # b= np.append(b,[[emg[0],emg[1],emg[2],emg[3],emg[4],emg[5],emg[6],emg[7]]],0)
            # if b.shape[0]==512:
            # final(b)

    def process_imu(self,quat, acc, gyro):
        print(quat)

    def process_sync(self,arm, x_direction):
        print(arm, x_direction)

    def process_classifier(self,pose):
        print(pose)

    def process_battery(self,batt):
        print("Battery level: %d" % batt)

    def led_emg(self,emg):
        if (emg[0] > 80):
            myo_device.services.set_leds( [255, 0, 0], [128, 128, 255] )
        else:
            myo_device.services.set_leds( [128, 128, 255], [128, 128, 255] )
    def ReadEMG(self):
        while self.stop_request :
            time.sleep(0.09)
            if self.myo_device.services.waitForNotifications( 1 ):
                continue
            print("Waiting...")

"""
Real = RealTime()
Real.start_MYO()
Real.stop_request = True
threading.Thread( target=Real.ReadEMG() ).start()
#time.sleep(3)
Real.stop_request = False
print ("Hi")



while True:
    if Real.myo_device.services.waitForNotifications(1):
		print (Real.EMG.shape[0])
		continue
	
        
        
    print("Waiting...")


"""





