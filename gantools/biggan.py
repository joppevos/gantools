# methods for setting up and interacting with biggan
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
from itertools import cycle
from pathlib import Path

#session = InteractiveSession(config=config)

school = '/home/linuxhku/projects/gantools/gantools/model'
MODULE_PATH = school

class BigGAN(object):
    def __init__(self, module_path=MODULE_PATH):
        tf.reset_default_graph()
        print('Loading BigGAN module from:', module_path)

        #-----------------------------------------------------------------
        # fix "RuntimeError: Exporting/importing meta graphs is not
        # supported when eager execution is enabled." error when importing
        # the tfhub module
        tf.disable_eager_execution()
        #-----------------------------------------------------------------
        module = hub.Module(module_path)
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                for k, v in module.get_input_info_dict().items()}
        self.input_z = self.inputs['z']
        self.dim_z = self.input_z.shape.as_list()[1]
        self.input_y = self.inputs['y']
        self.vocab_size = self.input_y.shape.as_list()[1] # dimension of y (aka label count)
        self.input_trunc = self.inputs['truncation']
        self.output = module(self.inputs)

        # initialize/instantiate tf variables
        initializer = tf.global_variables_initializer()

        #-----------------------------------------------------------------
        # fix "could not create cudnn handle" error
        # see: https://github.com/tensorflow/tensorflow/issues/24496
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #-----------------------------------------------------------------

        self.sess = tf.Session(config=config)
        self.sess.run(initializer)

    # NOTE: use save callback to save images once per batch. return type changes to None.
    def sample(self, vectors, labels, truncation=0.5, batch_size=1, save_callback=None):
        num = vectors.shape[0]

        # deal with scalar input case
        truncation = np.asarray(truncation)
        if truncation.ndim == 0:# truncation is a scalar
            #TODO: there has to be a better way to do this...
            truncation = cycle([truncation])

        ims = []
        for batch_start, trunc in zip(range(0, num, batch_size), truncation):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {self.input_z: vectors[s], self.input_y: labels[s], self.input_trunc: trunc}
            ims_batch = self.sess.run(self.output, feed_dict=feed_dict)
            ims_batch = np.clip(((ims_batch + 1) / 2.0) * 256, 0, 255)
            ims_batch = np.uint8(ims_batch)
            if save_callback is None:
                ims.append(ims_batch)
            else:
                save_callback(ims_batch)
        if save_callback is None:
            ims = np.concatenate(ims, axis=0)
            assert ims.shape[0] == num
            return ims
        else:
            return None


    # TODO: make a version of sample() that includes a callback function to save ims somewhere instead of keeping
    # them in memory.
