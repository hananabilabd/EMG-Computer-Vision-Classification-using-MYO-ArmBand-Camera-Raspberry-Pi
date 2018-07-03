
# coding: utf-8

# In[1]:


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





# In[10]:






# In[257]:


filename =  "/home/ub/Downloads/GP_Some_Statistical_Result/s2/S1_E1_A1"  #change the subject S1,S2,S3 .. S7(23%), S6(75%)
Data=upload_mat(filename)
Data['emg'][0].values.max()


# In[258]:


emg_df=Data['emg']
stimulus_df=Data["stimulus"]
rep_df=Data['repetition']

emg_df["label"]=stimulus_df
emg_df["rep"] =rep_df
emg_df=emg_df.set_index("label")
# emg_df
# #intended_movement_labels=np.random.randint(1,18,size=5)
# intended_movement_labels=[1,3,10,11,15] 
# #intended_movement_labels=[6,16,17,13,14] #62% acc
intended_movement_labels=[1,6,14,15,16,17]  #acc 75% at k=5.. acc 73% at k=7
emg_df=emg_df.loc[intended_movement_labels]

emg_df=emg_df.reset_index()  
emg_df
updated_label=emg_df['label']
updated_rep=emg_df['rep']
emg_df=emg_df.drop(['label','rep'],1)


# In[259]:


emg_df


# In[2]:


-1.163372e-06


# In[260]:


filtered_emg=filteration (emg_df)
mean,std,normalized_emg=mean_std_normalization (filtered_emg)
#normalized_emg=filteration (emg_df)

normalized_emg['label']=updated_label
normalized_emg['rep']=updated_rep

normalized_emg=normalized_emg.set_index('rep')
normalized_emg


# In[261]:


#the classifier, prepare train part
rep_train=[1,3,6,4]
normalized_emg_train,LL_train=prepare_df(rep_train,normalized_emg)
predictors_train,outcomes_train=get_predictors_and_outcomes(intended_movement_labels,rep_train,normalized_emg_train,LL_train)

#prepare test part
rep_test=[2,5]
normalized_emg_test,LL_test=prepare_df(rep_test,normalized_emg)
normalized_emg_test

# predictors_test,outcomes_test=get_predictors_and_outcomes(intended_movement_labels,rep_test,normalized_emg_test,LL_test)
# print len(predictors_test[0])
# outcomes_test


# In[262]:


#KNN classifier applied
k_compare = []
k_values=range(1,20)
for k in k_values:
    clf=KNeighborsClassifier(n_neighbors=k) 
    clf.fit(predictors_train,outcomes_train)

    #Accuracy of the test data
    accuracy_test=clf.score(predictors_test,outcomes_test)*100
    k_compare.append(accuracy_test)
    
k_df=pd.DataFrame(k_compare,index=k_values)
k_df=k_df.rename(columns={0:"Accuracy"})
best_k=np.argmax(k_df["Accuracy"])
KNN_accuracy_test = np.max(k_compare)
print("at K={} Accuracy is {}% ".format(best_k,KNN_accuracy_test))



# In[76]:


#SVM classifier applied

clf=svm.LinearSVC(dual=False) # at C= 0.05:0.09 gives little increase in accuracy, around 0.4%
#clf=svm.SVC(C=1,gamma=10000) #this is not good as it uses one VS one technique not one VS all
clf.fit(predictors_train,outcomes_train)

#Accuracy of the test data
SVM_accuracy_test=clf.score(predictors_test,outcomes_test)*100
print("SVM Accuracy is {}%".format(SVM_accuracy_test))


# In[79]:


KNN_accuracy_many_subjects =[81,74,85,68,30,66]
SVM_accuracy_many_subjects =[84,85,89,75,27,77]
subject_numbers=["S1","S2","S3","Amp_S6","Amp_S7","Amp_S11"]
visualization_KNN_vs_SVM_accuracy(subject_numbers,KNN_accuracy_many_subjects,SVM_accuracy_many_subjects)


# In[81]:


plt.figure()
visualization_K_accuracy (k_values,k_compare,best_k,subject="3")


# In[18]:


#test online part
#inside the while, i expect four main funcs. 1)windowing MES ..  2)extract features .. 3)classifier .. 4)post proc
width=512; tau=512 ; end=0; win_num=0; labels_online_total=[]; outcomes_online=[];
rep_online=[2,5]
normalized_emg_online,LL_online=prepare_df(rep_online,normalized_emg)

while((len(normalized_emg_online)-end) >= width):
    
    # first func. : Windowing MES
    end,window_df=MES_analysis_window(normalized_emg_online,width,tau,win_num)
    win_num=win_num + 1
    
    #second func. : extract features
    test_features_array=features_extraction(window_df)
    test_features_array=test_features_array.reshape(1,-1)
    
    
    #third func. : classifer and getting the decesion
    label_online=clf.predict(test_features_array)
    labels_online_total.append(label_online)
    mode,count=stats.mode(LL_online.iloc[win_num*tau: ((win_num*tau)+width)])
    outcomes_online.append(mode)
    
    #forth func. : post processing (majority vote for the result)
    

    


    
#to see is the post processing affect the accuracy or not?   
accuracy_online=np.mean((np.array(labels_online_total)==np.array(outcomes_online)))*100
accuracy_online


# In[19]:


visualization_decesions(labels_online_total)


# In[175]:


#Post processing trail 
#for i in range(0,len(labels_online_total),10):
#    mode,count=stats.mode(labels_online_total[i:i+10])
#    labels_online_total[i:i+10]=9*[mode]
#np.mean((np.array(labels_online_total)==np.array(outcomes_online)))*100


# In[8]:


#range for zooming  --> 0 : df.shape[0]/fs
visualization_all_ch(normalized_emg,fs=2000,zooming_from_to=(0,100))
#visualization_one_ch(normalized_emg,ch_num=5,fs=2000,zooming_from_to=(0,20))
#E_df=Env_try (normalized_emg,fs=2000,order=4)
#visualization_all_ch(emg_df)


# In[15]:


s= pd.read_csv('foo' +"5"+".csv" ,nrows =2000,header=None)
s['label'] = '1'
s.columns = [1,2,3,4,5,6,7,8,'label']
s


# In[10]:


rep = []

for i in range(1,7):
    for j in range(0,667):
        rep.append(i) 

rep = rep[:4000]
emg_set = {
    
}
intended_movement_labels = [5,6,14,15,0]

for i in intended_movement_labels:
  #  emg_set[i] =  pd.read_csv('Hanna/' +str(i)+".csv" ,nrows =10900)
   # emg_set[i] =  pd.read_csv('foo' +str(i)+".csv" ,nrows =10900)
    emg_set[i] = pd.read_csv('Hanna_Model/' +str(i)+".csv" ,nrows =4000,header=None)

    emg_set[i]['label'] = i
    emg_set[i].columns = [1,2,3,4,5,6,7,8,'label']
    emg_set[i]['rep'] = rep
    
# s = emg_set[6].combine(  emg_set[1])

data= pd.concat([emg_set[5],emg_set[6],emg_set[14] , emg_set[15] , emg_set[0] ])
data = data.drop_duplicates().reset_index(drop=True)
dataLabel=data['label']
dataRep=data['rep']
data=data.drop(['label','rep'],1)

# data=data.reset_index()  
normalized_emg=filteration (data,sample_rate=200)
#mean,std,normalized_emg=mean_std_normalization (filtered_emg)
#normalized_emg = normalized_emg.drop_duplicates().reset_index(drop=True)
normalized_emg['label'] = dataLabel

normalized_emg['rep'] = dataRep
normalized_emg


normalized_emg=normalized_emg.set_index('rep')
normalized_emg
rep_train=[1,3,6,4]
normalized_emg_train,LL_train=prepare_df(rep_train,normalized_emg)
predictors_train,outcomes_train=get_predictors_and_outcomes(intended_movement_labels,rep_train,normalized_emg_train,LL_train)

#prepare test part
rep_test=[2,5]
normalized_emg_test,LL_test=prepare_df(rep_test,normalized_emg)

#normalized_emg_test
predictors_test,outcomes_test=get_predictors_and_outcomes(intended_movement_labels,rep_test,normalized_emg_test,LL_test)
#print len(predictors_test[0])
#outcomes_test
predictors_testt = get_predictors(normalized_emg_test)
normalized_emg_test


# In[38]:


normalized_emg_test


# In[11]:


k_compare = []
k_values=range(1,20)
for k in k_values:
    clf_knn=KNeighborsClassifier(n_neighbors=k) 
    clf_knn.fit(predictors_train,outcomes_train)

    #Accuracy of the test data
    accuracy_test=clf_knn.score(predictors_test,outcomes_test)*100
    k_compare.append(accuracy_test)
    
k_df=pd.DataFrame(k_compare,index=k_values)
k_df=k_df.rename(columns={0:"Accuracy"})
best_k=np.argmax(k_df["Accuracy"])
KNN_accuracy_test = np.max(k_compare)
print("at K={} Accuracy is {}% ".format(best_k,KNN_accuracy_test))


# In[29]:


predictors_train[0].shape


# In[12]:



clf_svm=svm.LinearSVC(dual=False) # at C= 0.05:0.09 gives little increase in accuracy, around 0.4%
#clf=svm.SVC(C=1,gamma=10000) #this is not good as it uses one VS one technique not one VS all
clf_svm.fit(predictors_train,outcomes_train)

#Accuracy of the test data
SVM_accuracy_test=clf_svm.score(predictors_test,outcomes_test)*100
print("SVM Accuracy is {}%".format(SVM_accuracy_test))


# In[31]:


from sklearn.externals import joblib
filename = 'EMG_hannaa_model.pickle'
joblib.dump(clf_svm, filename)


# In[36]:


outcomes_test


# In[52]:


visualization_one_ch(emg_set[17],1,fs=200,zooming_from_to=(0,100))


# In[ ]:




