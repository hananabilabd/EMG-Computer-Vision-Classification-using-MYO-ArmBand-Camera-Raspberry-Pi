import h5py
import numpy as np
import random
import time
from scipy import misc
#import queue  ##If python 3
import Queue as queue ##If python 2
import threading
from keras.layers import Input, Add, Dense, Activation,Dropout , BatchNormalization, Flatten, Conv2D,MaxPooling2D
from keras.models import Model #, load_model
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf

class CV():
    def __init__(self, queue_size=8):
        self.q = queue.Queue()
        self.stage = 0
        self.corrections = 0
        self.all_grasps = [1, 2, 3, 4]
        self.Choose_grasp = list( self.all_grasps )
        self.graph = tf.get_default_graph()

    def rgb2gray(self,rgb_image):
        return np.dot( rgb_image, [0.299, 0.587, 0.114] )


    def real_preprocess(self,img):
        # gray level
        img_gray = self.rgb2gray( img )
        # resize the image 48x36:
        img_resize = misc.imresize( img_gray, (48, 36) )
        # Normalization:
        img_norm = (img_resize - img_resize.mean()) / img_resize.std()
        return img_norm


    def Nazarpour_model(self,input_shape, num_of_layers=2):
        x_input = Input( input_shape )
        x = Conv2D( 5, (5, 5), strides=(1, 1), padding='valid' )( x_input )
        x = BatchNormalization( axis=3 )( x )
        x = Activation( 'relu' )( x )
        x = Dropout( 0.2 )( x )
        if num_of_layers == 2:
            x = Conv2D( 25, (5, 5), strides=(1, 1), padding='valid' )( x )
            x = BatchNormalization( axis=3 )( x )
            x = Activation( 'relu' )( x )
        x = MaxPooling2D( (2, 2), strides=(2, 2) )( x )
        x = Dropout( 0.2 )( x )
        x = Flatten()( x )
        x = Dense( 4, activation='softmax', kernel_initializer=glorot_uniform( seed=0 ) )( x )
        model = Model( inputs=x_input, outputs=x )
        return model


    def grasp_type(self,path_of_test_real, model_name):
        """
        path_of_test_real : the path of the uploaded image in case of offline.
        model_name: the name of the trained model, 'tmp.h5'

        """

        n_row = 48
        n_col = 36
        nc = 1
        model = self.Nazarpour_model( (n_row, n_col, nc), num_of_layers=2 )
        model.compile( 'adam', loss='categorical_crossentropy', metrics=['accuracy'] )
        model.load_weights( self.path1 )

        i = misc.imread( self.model_name )
        img_after_preprocess = self.real_preprocess( i )
        x = np.expand_dims( img_after_preprocess, axis=0 )
        x = x.reshape( (1, n_row, n_col, nc) )
        out = model.predict( x )
        grasp = np.argmax( out ) + 1
        if grasp == 1 :
            print( ("Grasp_Type : Pinch  , Class = 1 \n ") )
        if grasp == 2 :
            print( ("Grasp_Type  : Palmar Wrist Neutral , Class = 2 \n ") )
        if grasp == 3 :
            print( ("Grasp_Type  : Tripod , Class = 3 \n ") )
        if grasp == 4 :
            print( ("Grasp_Type : Palmar Wrist Pronated,, Class = 4 \n ") )
        return grasp







    def Main_algorithm(self,path1,path2=None):

        self.path1 = path1  # put the path of the tested picture
        if path2:
            self.model_name = path2
        else:
            self.model_name = 'tools/class 1/50_r110.png'




        #    path_of_real_test='/home/ghadir/Downloads/__/class 1/50_r110.png' #put the path of the tested picture
        #    CV_model_name='GP_Weights.h5'
        # """

        while not (self.q.empty()):
            EMG_class_recieved = self.q.get()
            if (EMG_class_recieved == 1 or self.stage == 0):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage ))
                self.System_power( 1 )  # Start system

            elif (EMG_class_recieved == 1):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage ))
                self.Confirmation()

            elif (EMG_class_recieved == 2):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage ))
                self.Cancellation()

            elif (EMG_class_recieved == 0):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage ))
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
            with self.graph.as_default():
                self.grasp = self.grasp_type( self.path1, self.model_name )
            print('Preshaping grasp type {}\n\n').format( self.grasp )
            self.stage = 1

    def Confirmation(self):


        print("    Confirmed! \n")
        if self.stage < 2:
            self.stage += 1
            self.corrections = 0
            self.Choose_grasp = list( self.all_grasps )
            print("Grasping ... grasp type{} \n\n").format( self.grasp )
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
                    if self.grasp in self.Choose_grasp:
                        self.Choose_grasp.remove( self.grasp )
                if not self.Choose_grasp: #Check if list is empty after removing an element.
                    self.Choose_grasp = list( self.all_grasps )
                    self.corrections = 0
                self.grasp = random.SystemRandom().choice( self.Choose_grasp )
                print('preshaping grasp type {}\n\n').format( self.grasp )
                self.stage = 1
            else:
                # Redo previous action:
                if self.stage == 0:
                    self.System_power( 1 )
                    self.corrections += 1
                    print ("Restarting ... \n")
                elif self.stage == 1:
                    print('Preshaping grasp type {}\n\n').format( self.grasp )
                elif self.stage == 2:
                    print("Grasping ... grasp type{} \n\n").format( self.grasp )
            print ("Correction no. {}").format( self.corrections + 1 )


        else:
            print ('No previous stage, restarting ... \n')
            self.System_power( 1 )






            #q = queue.Queue()



"""
Stages meanings:
0: System off
1: Taking photos, deciding grasp type, preshaping.
2: Grasping
3: Releasing
"""

#cv =CV()



# t1 = threading.Thread(target = EMG_Listener, name ='thread1')
# t2 = threading.Thread(target = Main_algorithm, name ='thread2')

# t1.daemon = True
# t2.daemon = True

# t1.start()
# t2.start()

# t1.join()
#grasp = cv.grasp_type( 'tools/class 1/50_r110.png', 'tools/GP_Weights.h5' )
#print ('Grasp type no.{0} \n'.format( grasp ))

