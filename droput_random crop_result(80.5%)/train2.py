import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

os.environ["TFHUB_CACHE_DIR"] = './cache'
data_dir = './data'


module_selection = ("mobilenet_v2_140_224", 224)

handle_base, pixels = module_selection

MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)

IMAGE_SIZE = (pixels, pixels)

BATCH_SIZE = 128 #@param {type:"integer"}

print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))


datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")


do_data_augmentation = True #@param {type:"boolean"}

if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen



# training
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)

# validation
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)

valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)



do_fine_tuning = False #@param {type:"boolean"}

print("Building model with", MODULE_HANDLE)


# add a preprocessing layers （randomcrop)
random_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(224, 224)
])



model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
     # crop before inference time
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    random_crop,
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning), # pretraind model
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])


model.build((None,)+IMAGE_SIZE+(3,))
model.summary()


model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.99),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=[tf.keras.metrics.CategoricalAccuracy()])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size


# add tensorflow callbacks
callbacks = [tf.keras.callbacks.CSVLogger('./training.csv', separator=",", append=False), # save performance to csv
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2, patience=1, min_lr=1e-8, verbose=1, mode='auto'), # reduce lr if necessary
             tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoint',
                save_weights_only=True,
                monitor='val_acc',
                mode='max',
                save_best_only=True) # save checkpoints
            ]

# train the model
hist = model.fit(
    train_generator,
    epochs=10, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps, callbacks=callbacks)


saved_model_path = "./result"
tf.saved_model.save(model, saved_model_path)