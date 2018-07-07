from sklearn.externals import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter,lfilter,filtfilt
from scipy import stats
from sklearn import svm
class EMG_Model():

    def filteration (self,data,sample_rate=2000.0,cut_off=20.0,order=5,ftype='highpass'): 
        nyq = .5 * sample_rate
        b,a=butter(order,cut_off/nyq,btype=ftype)
        d= lfilter(b,a,data,axis=0)
        return pd.DataFrame(d)

    
    def mean_std_normalization (self,df):
        m = df.mean(axis=0)
        s =df.std(axis=0)
        normalized_df =df/m
        return m,s,normalized_df



    def MES_analysis_window (self,df,width,tau,win_num):
        df_2=pd.DataFrame()
        start= win_num*tau
        end= start+width
        df_2=df.iloc[start:end]
        return end,df_2



    def prepare_df(self,rep,normalized_emg):
        df=normalized_emg.loc[rep]
        df=df.reset_index()  
        LL=df['label']
        df=df.drop(['rep','label'],1)
        
        return df,LL

    def features_extraction (self,df,th=0):
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

    def get_predictors_and_outcomes(self,intended_movement_labels,rep,emg,label_series,width=512,tau=128):
        
        x=[];y=[];
        end=0; win_num=0; 
        while((len(emg)-end) >= width):
            end,window_df=self.MES_analysis_window(emg,width,tau,win_num)
            win_num=win_num + 1
            
            ff=self.features_extraction(window_df)
            x.append(ff)
            
            expected_labels=label_series.iloc[win_num*tau: ((win_num*tau)+width)]
            mode,count=stats.mode(expected_labels)
            y.append(mode)
            
        predictors_array=np.array(x)
        outcomes_array=np.array(y)

        nsamples, nx, ny = predictors_array.shape
        predictors_array_2d = predictors_array.reshape((nsamples,nx*ny))

        return np.nan_to_num(predictors_array_2d),np.nan_to_num(outcomes_array)


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

    def prepare_data(self,intended_movement_labels=[0,1,2,3],rows=8000):
        emg_set = {}

        emg_set[0] = pd.read_csv( self.path1, header=None )
        emg_set[1] = pd.read_csv( self.path2, header=None )
        emg_set[2] = pd.read_csv( self.path3, header=None )
        emg_set[3] = pd.read_csv( self.path4, header=None )
        rows = min( emg_set[0].shape[0], emg_set[1].shape[0], emg_set[2].shape[0], emg_set[3].shape[0] )

        rep = []
        reps =rows // 6 if rows % 6 == 0 else (rows //6)+1

        for i in range(1,7):
            for j in range(0,reps):
                rep.append(i)

        rep = rep[:rows]


        for i in intended_movement_labels:
            #emg_set[i] = pd.read_csv('models/' +str(i)+".csv" ,nrows =rows,header=None)
            emg_set[i]['label'] = i
            emg_set[i].columns = [1,2,3,4,5,6,7,8,'label']
            emg_set[i]['rep'] = rep

        data = pd.DataFrame()

        for i in intended_movement_labels:
            data = pd.concat([data,emg_set[i]])

        data = data.drop_duplicates().reset_index(drop=True)
        dataLabel=data['label']
        dataRep=data['rep']
        data=data.drop(['label','rep'],1)

        normalized_emg=self.filteration (data,sample_rate=200)

        normalized_emg['label'] = dataLabel

        normalized_emg['rep'] = dataRep
        normalized_emg=normalized_emg.set_index('rep')
        rep_train=[1,3,6,4]
        normalized_emg_train,LL_train=self.prepare_df(rep_train,normalized_emg)
        predictors_train,outcomes_train=self.get_predictors_and_outcomes(intended_movement_labels,rep_train,normalized_emg_train,LL_train)

        #prepare test part
        rep_test=[2,5]
        normalized_emg_test,LL_test=self.prepare_df(rep_test,normalized_emg)

        #normalized_emg_test
        predictors_test,outcomes_test=self.get_predictors_and_outcomes(intended_movement_labels,rep_test,normalized_emg_test,LL_test)

        predictors_test = self.get_predictors(normalized_emg_test)
        return predictors_train,outcomes_train,predictors_test,outcomes_test

    def svm_model(self,predictors_train,outcomes_train):

        model=svm.LinearSVC(dual=False) # at C= 0.05:0.09 gives little increase in accuracy, around 0.4%
        model.fit(predictors_train,outcomes_train)
        return model

    def accuracy(self,model):
        return model.score(self.predictors_test,self.outcomes_test)*100

    def save_model(self,model,filename):
        joblib.dump(model, filename)



    def all_steps(self,path1,path2,path3,path4,file_name,movements=[0,1,2,3]):
        self.path1=path1
        self.path2=path2
        self.path3=path3
        self.path4=path4
        predictors_train,outcomes_train,self.predictors_test,self.outcomes_test = self.prepare_data(movements)
        model = self.svm_model(predictors_train,outcomes_train)

        #if you wanna accuracy
        print (self.accuracy(model))

        #save pickle
        self.save_model(model,file_name)

if __name__ == '__main__':
    e=EMG_Model()
    e.all_steps(path1="0.csv",path2="1.csv",path3="2.csv",path4="3.csv",file_name="Hannon.pickle")

