import threading
import RealTime
import poweroff
import numpy as np
#EMG ############################################################################################################
## All you have to do is to call start_thread1 to start threading smoothly
Real = RealTime.RealTime()

# self.Real.set_GP_instance(self)
Power = poweroff.poweroff()
thread1 = None
thread2 = None
event_stop_thread1 = threading.Event()
event_stop_thread2 = threading.Event()


def start_thread1():
    Real.Flag_Predict = True
    Real.b = np.empty( [0, 8] )
    event_stop_thread1.clear()
    thread1 = threading.Thread( target=loop1 )
    thread1.start()



def loop1():
    while not event_stop_thread1.is_set():
        # if not self.stop_threads.is_set():
        if Real.myo_device.services.waitForNotifications( 1 ):
            print(Real.predictions_array)
           
        else:
            print("Waiting...")



def stop_thread1():
    event_stop_thread1.set()
    thread1.join()
    thread1 = None
    Real.b = np.empty( [0, 8] )
    Real.Flag_Predict = False
Real.start_MYO()
start_thread1()

###########################################################################################################3
