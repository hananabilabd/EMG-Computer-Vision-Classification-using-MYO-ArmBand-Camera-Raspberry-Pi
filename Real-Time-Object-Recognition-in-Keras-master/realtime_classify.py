#==============================================================================
# REAL LIFE CLASSIFICATION
#==============================================================================
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.applications import VGG16
from keras.applications import ResNet50
import cv2, threading
import numpy as np
import time

# Initialize global variables to be used by the classification thread
# and load up the network and save it as a tensorflow graph

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.label = ''
        self.frame_to_predict = None
        self.classification = True
        self.model = ResNet50( weights='imagenet' )
        self.graph = tf.get_default_graph()
        self.score = .0
        print( 'Loading network...' )
        # self.model = VGG16(weights='imagenet')
        self.model = ResNet50( weights='imagenet' )
        self.graph = tf.get_default_graph()
        print( 'Network loaded successfully!' )

    def run(self):
        
        with self.graph.as_default():
        
            while self.classification is True:
                if self.frame_to_predict is not None:
                    self.frame_to_predict = cv2.cvtColor(self.frame_to_predict, cv2.COLOR_BGR2RGB).astype(np.float32)
                    self.frame_to_predict = self.frame_to_predict.reshape((1, ) + self.frame_to_predict.shape)
                    self.frame_to_predict = imagenet_utils.preprocess_input(self.frame_to_predict)
                    predictions = self.model.predict(self.frame_to_predict)
                    (self.imageID, self.label, self.score) = imagenet_utils.decode_predictions(predictions)[0][0]
                    print ((self.label ,self.score))
    
    def run_camera(self):
        # Initialize OpenCV video captue
        self.video_capture = cv2.VideoCapture( 0 )  # Set to 1 for front camera
        self.video_capture.set( 4, 800 )  # Width
        self.video_capture.set( 5, 600 )  # Height

        # Start the video capture loop
        while (True):

            # Get the original frame from video capture
            ret, original_frame = self.video_capture.read()
            # Resize the frame to fit the imageNet default input size
            self.frame_to_predict = cv2.resize( original_frame, (224, 224) )

            # Add text label and network score to the video captue
            cv2.putText( original_frame, "Label: %s | Score: %.2f" % (self.label, self.score),
                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2 )
            # Display the video
            cv2.imshow( "Classification", original_frame )

            # Hit q or esc key to exit
            if (cv2.waitKey( 1 ) & 0xFF == ord( 'q' )):
                break;

    def close(self):
        self.classification = False
        self.video_capture.release()
        cv2.destroyAllWindows()

# Start a keras thread which will classify the frame returned by openCV
#keras_thread = MyThread()
#keras_thread.start()
#keras_thread.run_camera()
#time.sleep(2)
#keras_thread.close()


        

