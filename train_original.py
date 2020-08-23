from dotenv import load_dotenv
import itertools
import os
import signal
import requests

from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import coremltools as ct
import collections
import re
import subprocess
import shutil

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


load_dotenv()

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


##
# Setup and select Hub module for retraining
##

module_selection = ("mobilenet_v2_100_224", 224)
do_fine_tuning = False
do_data_augmentation = False
saved_model_path = "./output/saved_model"
saved_labels_path = saved_model_path + "/saved_labels.txt"

handle_base, pixels = module_selection
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(
    handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32
EPOCHS = 5


##
# Setup image generator and preprocessing for dataset
##

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                    directory='data',
                                                    shuffle=False,
                                                    target_size=IMAGE_SIZE,
                                                    subset="training",
                                                    class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory='data',
                                                         shuffle=True,
                                                         target_size=IMAGE_SIZE,
                                                         subset="validation",
                                                         class_mode='categorical')


##
# Build the model
##

print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_dataset.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])

steps_per_epoch = train_dataset.samples // train_dataset.batch_size
validation_steps = validation_dataset.samples // validation_dataset.batch_size


def convert_to_tflite():
    optimize_lite_model = True
    num_calibration_examples = 60
    representative_dataset = None

    if optimize_lite_model and num_calibration_examples:
        def representative_dataset(): return itertools.islice(
            ([image[None, ...]]
             for batch, _ in train_dataset for image in batch),
            num_calibration_examples)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        if optimize_lite_model:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_dataset:
            converter.representative_dataset = representative_dataset
        lite_model_content = converter.convert()

        with open(saved_model_path + "/model.lite", "wb") as f:
            f.write(lite_model_content)
            print("Wrote %sTFLite model of %d bytes." %
                  ("optimized " if optimize_lite_model else "", len(lite_model_content)))


def convert_to_core_ml():
    image_input = ct.ImageType(shape=(1, 224, 224, 3,),
                               bias=[-1, -1, -1], scale=1/127)
    classifier_config = ct.ClassifierConfig(saved_labels_path)

    ml_model = ct.convert(
        model, inputs=[image_input], classifier_config=classifier_config,
    )

    ml_model.save(saved_model_path + "/model.mlmodel")


def convert_to_tfjs():
    subprocess.check_output(
        ['tensorflowjs_converter', '--input_format=tf_saved_model', '--output_node_names=final_result', saved_model_path, saved_model_path + "/web_model"])


def generate_labels():
    image_dir = "./data"
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in os.walk(image_dir))
    is_root_dir = True

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = sorted(set(os.path.normcase(ext)
                                for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
        file_list = []
        dir_name = os.path.basename(
            sub_dir[:-1] if sub_dir.endswith('/') else sub_dir)

        if dir_name == image_dir:
            continue

        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        training_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images
        }

    with open(saved_labels_path, 'w') as f:
        f.write('\n'.join(result.keys()) + '\n')


##
# Train the model
##
request_headers = {"Authorization": "Bearer " +
                   os.getenv("API_AUTHENTICATION_KEY")}


try:
    hist = model.fit(
        train_dataset,
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps
    )

    # ##
    # # Save and post the model to the back-end
    # ##

    tf.saved_model.save(model, saved_model_path)

    generate_labels()
    convert_to_tflite()
    convert_to_core_ml()
    convert_to_tfjs()

    js_zip_path = shutil.make_archive(
        'web_model', 'tar', saved_model_path + "/web_model")

    files = {
        'labels': open(saved_labels_path, 'rb'),
        'ml': open(saved_model_path + "/model.mlmodel", 'rb'),
        'tflite': open(saved_model_path + "/model.lite", 'rb'),
        'js': open(js_zip_path, 'rb'),
    }

    os.remove(js_zip_path)

    requests.post(os.getenv("API_URL") + "/webhooks/training-succeeded",
                  headers=request_headers,
                  files=files, verify=False)

except BaseException as error:
    print(error)

    requests.post(os.getenv("API_URL") + "/webhooks/training-failed", headers=request_headers,
                  data={"error": error}, verify=False)
