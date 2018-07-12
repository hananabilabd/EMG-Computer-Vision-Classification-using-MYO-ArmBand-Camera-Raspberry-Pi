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
import random
import queue  ##If python 3
# import Queue as queue ##If python 2
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
        ###
        self.q = queue.Queue()
        self.stage = 0
        self.corrections = 0
        self.all_grasps = [1, 2, 3, 4]
        self.Choose_grasp = list( self.all_grasps )
        self.grasp1 = None
        ###
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
                    #print ((self.label ,self.score))
                if self.classification == False :
                    break;
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

    def grasp_type(self):
        if self.label == "cellular_phone" :
            grasp =1
        elif self.label == "sunglasses" :
            grasp =2
        elif self.label == "beer_glass" :
            grasp =3
        elif self.label == "wallet" or "coffee_mug" or "water_bottle" or "revolver" :  ##
            grasp =4
            print (("z3bolaaaaaaaaaaaaaaaaaaaaa"))
        else :
            print (("Not Defined Grasp"))
            grasp =5
        return grasp


    def Main_algorithm(self):

        while not (self.q.empty()):
            EMG_class_recieved = self.q.get()
            if (EMG_class_recieved == 0 or self.stage == 0):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.System_power( 1 )  # Start system

            elif (EMG_class_recieved == 1):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.Confirmation()

            elif (EMG_class_recieved == 2):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.Cancellation()

            elif (EMG_class_recieved == 3):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.System_power( 0 )  # Turn system off

    def System_power(self,Turn_on):



        # Reset values:
        self.stage = 0
        #    corrections= 0
        self.Choose_grasp = list( self.all_grasps )

        if not Turn_on:
            self.corrections = 0
            # Turn off
            print ("Turning off ... back to rest state. \n\n\n")
        else:
            # Start/restart
            self.grasp1 = self.grasp_type( )

            print ((self.grasp1))
            print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
            self.stage = 1

    def Confirmation(self):


        print("    Confirmed! \n")
        if self.stage < 2:
            self.stage += 1
            self.corrections = 0
            self.Choose_grasp = list( self.all_grasps )
            print(("Grasping ... grasp type{} \n\n").format( self.grasp1 ))
            # Do the action
        else:
            print ('Releasing ... \n')
            self.System_power( 0 )


    def Cancellation(self):



        if self.stage > 0:
            print("    Cancelled! \n")
            self.stage -= 1
            #        corrections +=1
            if (self.stage == 0 and self.corrections > 3):
                print("Exceeded maximum iteration: \n Choosing from remaining grasps")
                if self.Choose_grasp:
                    if self.grasp1 in self.Choose_grasp:
                        self.Choose_grasp.remove( self.grasp1 )
                if not self.Choose_grasp: #To check if list is empty after removing an element.
                    self.Choose_grasp = list( self.all_grasps )
                    self.corrections = 0
                self.grasp1 = random.SystemRandom().choice( self.Choose_grasp )
                print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
                self.stage = 1
            else:
                # Redo previous action:
                if self.stage == 0:
                    self.System_power( 1 )
                    self.corrections += 1
                    print ("Restarting ... \n")
                elif self.stage == 1:
                    print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
                elif self.stage == 2:
                    print(("Grasping ... grasp type{} \n\n").format( self.grasp1 ))
            print(("Correction no. {}").format( self.corrections + 1 ))


        else:
            print ('No previous stage, restarting ... \n')
            self.System_power( 1 )

# Start a keras thread which will classify the frame returned by openCV
#keras_thread = MyThread()
#keras_thread.start()
#keras_thread.run_camera()
#time.sleep(2)
#keras_thread.close()


        

