import open_myo as myo
import numpy as np
import sys

b= np.empty([0,8])
i=0
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
    ## for RAW_EMG 
    b = np.append(b,emg,axis =0)
    print (b)
    if b.shape[0]==512:
		final(b)
    
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
x =0
while x <10:
    x += 1
    if myo_device.services.waitForNotifications(1):
        continue
    print("Waiting...")
