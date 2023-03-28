# -*- coding: utf-8 -*-
"""


"""

import os
import tensorflow as tf
import tensorflow.keras.backend as KTF

from tensorflow.keras import backend as K
from tensorflow.keras import __version__

import warnings

warnings.filterwarnings('ignore')


def set_gpu():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # utilizing gpus #0 and #1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # utilizing gpus #0 and #1
    # tf.compat.v1.ConfigProto()
    # config = tf.ConfigProto()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess = tf.compat.v1.Session(config=config)

    # KTF.set_session(sess)
    tf.compat.v1.keras.backend.set_session(sess)


# if __name__ == '__main__':
def begin():
    set_gpu()
    print('GPU setting completed.')
    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_data_format())
