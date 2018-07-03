
# coding: utf-8

# In[ ]:


# for memory error >> try to change the float 64 to float 32 
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.signal import freqz
from scipy.signal import butter,lfilter,filtfilt
#from scipy.interpolate import interp1d
#from __future__ import division, print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
#from __future__ import print_function
import sys
import test
import time
import numpy as np
#from test import MyoRaw
import open_myo as myo

def upload_mat (filename):
    Data= sio.loadmat(filename)
    skips=[ '__header__','__globals__','__version__']
    for k in Data.keys():
        if k in skips:
            Data.pop(k,None)
        else:
            Data[k]=pd.DataFrame(Data[k])
    return (Data)



#search on Hampel filter to remove spikes. and make notch filter on 50 hz
def filteration (data,sample_rate=2000.0,cut_off=20.0,order=5,ftype='highpass'): 
    nyq = .5 * sample_rate
    b,a=butter(order,cut_off/nyq,btype=ftype)
    d= lfilter(b,a,data,axis=0)
    return pd.DataFrame(d)

 
def mean_std_normalization (df):
    m = df.mean(axis=0)
    s =df.std(axis=0)
    normalized_df =df/m
    return m,s,normalized_df



def MES_analysis_window (df,width,tau,win_num):
    df_2=pd.DataFrame()
    start= win_num*tau
    end= start+width
    df_2=df.iloc[start:end]
    return end,df_2



def prepare_df(rep,normalized_emg):
    df=normalized_emg.loc[rep]
    df=df.reset_index()  
    LL=df['label']
    df=df.drop(['rep','label'],1)
    
    return df,LL

def features_extraction (df,th=0):
    #F1 : mean absolute value (MAV)
    MAV=abs(df.mean(axis=0)) 
    
    MAV=list(MAV)
    WL = []
    SSC= []
    ZC = []
    for col,series in df.iteritems():
        #F2 : wave length (WL)
        s=abs(np.array(series.iloc[:-1])- np.array(series.iloc[1:]))
        WL_result=np.sum(s)
        WL.append( WL_result)
        
        #F3 : zero crossing(ZC)
        _1starray=np.array(series.iloc[:-1])
        _2ndarray=np.array(series.iloc[1:])
        ZC.append(((_1starray*_2ndarray<0) & (abs(_1starray - _2ndarray)>=th) ).sum())
        
        
         #F4 : slope sign change(SSC)
        _1st=np.array(series.iloc[:-2])
        _2nd=np.array(series.iloc[1:-1])
        _3rd=np.array(series.iloc[2:])
        SSC.append(((((_2nd - _1st)*(_2nd - _3rd))>0) &(((abs(_2nd - _1st))>=th) | ((abs(_2nd - _3rd))>=th))).sum())
    
    features_array=np.array([MAV,WL,ZC,SSC]).T
    return features_array

def get_predictors_and_outcomes(intended_movement_labels,rep,emg,label_series,width=512,tau=128):
    
    x=[];y=[];
    end=0; win_num=0; 
    while((len(emg)-end) >= width):
        end,window_df=MES_analysis_window(emg,width,tau,win_num)
        win_num=win_num + 1
        
        ff=features_extraction(window_df)
        x.append(ff)
        
        expected_labels=label_series.iloc[win_num*tau: ((win_num*tau)+width)]
        mode,count=stats.mode(expected_labels)
        y.append(mode)
        
    predictors_array=np.array(x)
    outcomes_array=np.array(y)

    nsamples, nx, ny = predictors_array.shape
    predictors_array_2d = predictors_array.reshape((nsamples,nx*ny))

    return np.nan_to_num(predictors_array_2d),np.nan_to_num(outcomes_array)


def get_predictors(emg,width=512,tau=128):
    
    x=[];
    end=0; win_num=0; 
    while((len(emg)-end) >= width):
        end,window_df=MES_analysis_window(emg,width,tau,win_num)
        win_num=win_num + 1
        
        ff=features_extraction(window_df)
        x.append(ff)
        
    predictors_array=np.array(x)

    nsamples, nx, ny = predictors_array.shape
    predictors_array_2d = predictors_array.reshape((nsamples,nx*ny))

    return np.nan_to_num(predictors_array_2d)

def visualization_decesions(labels_online_total):
    
    C_y_axis=labels_online_total
    C_x_axis=range(len(labels_online_total))
    plt.plot(C_x_axis,C_y_axis,'.k')
    plt.xlabel("Time")
    plt.ylabel("Class")
    plt.show()
    
def visualization_KNN_vs_SVM_accuracy(subject_numbers,KNN_accuracy_many_subjects,SVM_accuracy_many_subjects):
    
    get_ipython().magic(u'matplotlib notebook')
    xvals=range(len(KNN_accuracy_many_subjects))
    plt.bar(xvals,KNN_accuracy_many_subjects,width=0.3,label="KNN")

    new_xvals = []
    for i in xvals:
        new_xvals.append(i+0.3)

    plt.bar(new_xvals,SVM_accuracy_many_subjects,label="SVM",width=0.3,color="green")
    plt.legend()
    plt.xlim(0,(len(KNN_accuracy_many_subjects)+2))
    plt.ylim(0,100)
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison: KNN vs SVM")
    plt.xticks(new_xvals,subject_numbers,rotation=-30)
    plt.tick_params(top='off', bottom='off', left='on', right='off', labelleft='on', labelbottom='on')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()
    
    
def visualization_K_accuracy (x,y,best_k,subject):
    
    bars=plt.bar(x,y,align="center",color="lightslategrey",linewidth=0)
    bars[best_k-1].set_color('#1F77B4')
    plt.xticks(x,k_values,alpha=0.8)
    plt.title("KNN Accuracy of subject {} for different K values".format(subject))
    plt.tick_params(top='off', bottom='off', left='on', right='off', labelleft='on', labelbottom='on')
    plt.ylabel("Accuracy")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    bar=bars[best_k-1]
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())), 
                     ha='center', color='w', fontsize=12)
    plt.show()
    
   
def visualization_all_ch (df,zooming_from_to,fs=2000): #439 not fixed  : df.shape[0]/fs

    x=np.linspace(0,np.ceil(df.shape[0])/fs,df.shape[0]) #the time using sampling freq
    f, ax = plt.subplots(12, sharex=True,figsize=(25,25))
    for i in range(12):
        y = df.iloc[:,i]
        ax[i].plot(x,y)
        
    # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
    f.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim(zooming_from_to)
    plt.grid(True)
    plt.show()
    

    
    
def visualization_one_ch (df,ch_num,zooming_from_to,fs=2000):

    x=np.linspace(0,np.ceil(df.shape[0])/fs,df.shape[0]) #the time using sampling freq
    y=df.iloc[:,ch_num]#rows value in the intended electrode number
    plt.plot(x,y)
    plt.xlim(zooming_from_to)
    plt.grid(True)
    plt.show()

    


# In[ ]:

emg_total =  np.empty([0,8])
iteration = 0
def predict(emg,tau=128):
    #emg = np.random.rand(512,8)
    global b,emg_total,iteration
    emg_total= np.append(emg_total,b,axis=0)
    data= pd.DataFrame(emg_total)
    filtered_emg=filteration (data,sample_rate=200)
    predictors_test = get_predictors(filtered_emg)
    emg_total = emg_total[(iteration*tau):]
    iteration = iteration + 1
    b= np.empty([0,8])
    filename = 'EMG_hanna_model.pickle'
    pickled_clf=joblib.load(filename)
    return pickled_clf.predict(predictors_test)




# In[ ]:


b= np.empty([0,8])
p = np.empty([0])
predictions_array = []
###This the function you will receive your EMG data in ==> You can then thread it to do something else 
def final (emg):
	print (":D")
	
	print (emg.shape)
	#print emg[:,0] ## if you want a single channel
	global b 
	b= np.empty([0,8])
	
def process_emg(emg):
	#unfortunately the Filtered Array provide 1 array of 8 element at a time  ==> in te Form of Tuple 
	# while The RAW_EMG provide 2 array at a time 8 elements each , ===> in the form of list that contains 2 tuples 
	
    #print(emg)
    global b
    global p
    ## for RAW_EMG 
    b = np.append(b,emg,axis =0)
    if b.shape[0]==512:
        #final(b)
        p = np.append(p,predict(b), axis=0)
        #c=predict(b)
        #print (c)
        #print (predict(b))
        #predictions_array.append(predict(b))
    
    ## For Filtered_EMG
    #b= np.append(b,[[emg[0],emg[1],emg[2],emg[3],emg[4],emg[5],emg[6],emg[7]]],0)
    #if b.shape[0]==512:
        #final(b)
def process_imu(quat, acc, gyro):
    print(quat)

def process_sync(arm, x_direction):
    print(arm, x_direction)

def process_classifier(pose):
    print(pose)

def process_battery(batt):
    print("Battery level: %d" % batt)

def led_emg(emg):
    if(emg[0] > 80):
        myo_device.services.set_leds([255, 0, 0], [128, 128, 255])
    else:
        myo_device.services.set_leds([128, 128, 255], [128, 128, 255])
myo_mac_addr = myo.get_myo()
print("MAC address: %s" % myo_mac_addr)
myo_device = myo.Device()
myo_device.services.sleep_mode(1) # never sleep
myo_device.services.set_leds([128, 128, 255], [128, 128, 255])  # purple logo and bar LEDs)
myo_device.services.vibrate(1) # short vibration
fw = myo_device.services.firmware()
print("Firmware version: %d.%d.%d.%d" % (fw[0], fw[1], fw[2], fw[3]))
batt = myo_device.services.battery()
print("Battery level: %d" % batt)
#myo_device.services.emg_filt_notifications()
myo_device.services.emg_raw_notifications()
#myo_device.services.imu_notifications()
#myo_device.services.classifier_notifications()
# myo_device.services.battery_notifications()
myo_device.services.set_mode(myo.EmgMode.RAW, myo.ImuMode.OFF, myo.ClassifierMode.OFF)
myo_device.add_emg_event_handler(process_emg)
#myo_device.add_emg_event_handler(led_emg)
# myo_device.add_imu_event_handler(process_imu)
#myo_device.add_sync_event_handler(process_sync)
# myo_device.add_classifier_event_hanlder(process_classifier)

while True:
    if myo_device.services.waitForNotifications(1):
		print(p)
		continue
	
        
        
    print("Waiting...")






