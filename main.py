import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras

from data import *
from model import *

print("Use GPU ? ", tf.test.is_built_with_cuda())

config = tf.ConfigProto(device_count={'GPU': 0})
#config.gpu_options.allow_growth = True  # Allow GPU memory to grow, otherwise memory overflow exception
#run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

print('keras version is ', keras.__version__)
print('tensorflow version is ', tf.__version__)


session = tf.Session(config=config)
#tf.keras.backend.set_session(session) # not used in following
keras.backend.tensorflow_backend.set_session(session) # used in following


data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)
