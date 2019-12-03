#!/usr/bin/env python
# tensorflowapp ds app
#
# (c) 2016 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import keras.models as models
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import os
K.set_image_data_format('channels_last')
import cv2
import sys
# import the Chris app superclass
from chrisapp.base import ChrisApp



class Tensorflowapp(ChrisApp):
    """
    test tf apps.
    """
    AUTHORS         = 'FNNDSC (dev@babyMRI.org)'
    SELFPATH        = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC        = os.path.basename(__file__)
    EXECSHELL       = 'python3'
    TITLE           = 'tf training sample'
    CATEGORY        = 'tensorflow'
    TYPE            = 'ds'
    DESCRIPTION     = 'Sample Tensorflow training application plugin for ChRIS Project.'
    DOCUMENTATION   = 'http://wiki'
    VERSION         = '1'
    ICON            = '' # url of an icon image
    LICENSE         = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS = 1  # Override with integer value
    MAX_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT         = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT         = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Fill out this with key-value output descriptive info (such as an output file path
    # relative to the output dir) that you want to save to the output meta file when
    # called with the --saveoutputmeta flag
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        """
        self.add_argument('--prefix', dest='prefix', type=str, optional=True,
                          help='prefix for file names')
        self.add_argument('--inference_path', dest='inference_path', type=str,
                          optional=True, help='path of images')
        self.add_argument('--saved_model_name', dest='saved_model_name',
                          type=str, optional=True,
                          help='name for exporting saved model')
        self.add_argument("--run_mode",dest="run_mode",type=str,optional=False,help="Select run mode from train or infer")

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        if options.run_mode == "train":
            self.run_tensorflow_app(options)
        else:
            self.predict(options)

    def run_tensorflow_app(self, options):




        digit_image = None
        if options.inference_path:
            str_path = os.path.abspath(options.inference_path)
            
            print("Test Image shape: ", digit_image.shape)
        self.mrinet_training(options, digit_image)



    def get_label_data(self,options):
        label_data = np.ndarray((256,256,256),dtype=np.uint8)
        label_files = os.listdir(options.inputdir +"/label_images")
        for i in label_files:
            np.append(label_data,cv2.imread(options.inputdir + "/" + i))
        return label_data        

    def get_train_data(self,options):
        train_data = np.ndarray((256,256,256),dtype=np.uint8)
        in_files = os.listdir(options.inputdir + "/input_images")
        for i in in_files:
            np.append(train_data,cv2.imread(options.inputdir + "/" + i))
        return train_data

    def get_unet(self):
        inputs = Input((256, 256, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()
        #plot_model(model, to_file='model.png')

        model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def mrinet_training(self, options, digit_image):
        model = self.get_unet()

        print("Currently running as User ID: %s " % os.getuid())
        print("Trying to read from the directory %s " % options.inputdir)
        if os.path.isdir(options.inputdir):
            print("%s is a directory" % options.inputdir)
        else:
            print("%s is not a directory" % options.inputdir)

        print("Trying to read data from the directory %s " % (options.inputdir + "/input_images"))
       
        train_data = self.get_train_data(options)
        print("Loaded "+ str(len(train_data)) + " images")

        print("Trying to read labels")
        label_data = self.get_label_data(options)

        print("Loaded "+str(len(label_data)) + " images")
        train_data = np.expand_dims(train_data,axis=3)
        label_data = np.expand_dims(label_data,axis=3) 

        model.fit(train_data,label_data,epochs=1,batch_size=1,verbose=1,shuffle=True,validation_split=0.8)
        str_outpath = os.path.join(options.outputdir, options.saved_model_name, self.VERSION)
        if os.path.isdir(str_outpath):
                model.save(str_outpath)
        else:
            model.save(options.outputdir + "/model.h5")



    def create_output(self, options, key, value):
        new_name = options.prefix + key
        str_outpath = os.path.join(options.outputdir, new_name)
        str_outpath = os.path.abspath(str_outpath)
        print('Creating new file... %s' % str_outpath)
        if not os.path.exists(options.outputdir):
            try:
                os.mkdir(options.outputdir)
            except OSError:
                print("Creation of the directory %s failed" % options.outputdir)
            else:
                print("Successfully created the directory %s " % options.outputdir)
        with open(str_outpath, 'w') as f:
            f.write(str(value))

    def get_test_data(self,options):
        test_data = np.ndarray((1,256,256),dtype=np.uint8)
        test_files = os.listdir(options.inputdir +"/test_images")
        for i in test_files:
            np.append(test_data,cv2.imread(options.inputdir + "/" + i))
        return test_data

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        self.run_tensorflow_app(options)



    def predict(self,options):
        model = self.get_unet()
        model = load_model(options.outputdir + "/model.h5")
        test_data = self.get_test_data(options)
        test_data = np.expand_dims(test_data,axis=3)
        cv2.imwrite(options.outputdir + "/inference_image.jpg",model.predict(test_data))
        print("in predict method")





    
    def create_output(self, options, key, value):
        new_name = options.prefix + key
        str_outpath = os.path.join(options.outputdir, new_name)
        str_outpath = os.path.abspath(str_outpath)
        print('Creating new file... %s' % str_outpath)
        if not os.path.exists(options.outputdir):
            try:
                os.mkdir(options.outputdir)
            except OSError:
                print("Creation of the directory %s failed" % options.outputdir)
            else:
                print("Successfully created the directory %s " % options.outputdir)
        with open(str_outpath, 'w') as f:
            f.write(str(value))

    def load_graph(frozen_graph_filename, name_prefix="prefix"):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and return it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name=name_prefix)
        return graph

# ENTRYPOINT
if __name__ == "__main__":
    app = Tensorflowapp()
    app.launch()
