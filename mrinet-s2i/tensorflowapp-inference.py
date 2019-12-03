#!/usr/bin/env python3                                                       _
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

# import the Chris app superclass
from chrisapp.base import ChrisApp


class Tensorflowapp(ChrisApp):
    """
    test tf apps.
    """
    AUTHORS         = 'BillRainford (brain@redhat.com)'
    SELFPATH        = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC        = os.path.basename(__file__)
    EXECSHELL       = 'python3'
    TITLE           = 'tf inference sample'
    CATEGORY        = 'tensorflow'
    TYPE            = 'ds'
    DESCRIPTION     = 'Sample Tensorflow inference application plugin for ChRIS Project.'
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

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        """
        self.add_argument('--prefix', dest='prefix', type=str, optional=False,
                          help='prefix for file names')
        self.add_argument('--inference_path', dest='inference_path', type=str,
                          optional=False, help='path of images')
        self.add_argument("--saved_model_name", dest='saved_model_name',
                          type=str, optional=False,
                          help="Saved model file to import")
        self.add_argument("--model_base_path", dest='model_base_path',
                          type=str, optional=False,
                          help="Saved model file base path")

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

    def run_tensorflow_app(self, options):
        self.predict(options)

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
