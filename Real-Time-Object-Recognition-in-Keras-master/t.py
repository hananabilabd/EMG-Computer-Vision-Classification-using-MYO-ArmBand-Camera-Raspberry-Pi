import realtime_classify
import cv2, threading

keras_thread = realtime_classify.MyThread()
keras_thread.start()
keras_thread.run_camera()
#time.sleep(2)
#keras_thread.close()