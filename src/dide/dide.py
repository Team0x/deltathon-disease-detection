import numpy as np
import pickle
import cv2
import os
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils.image_utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split




# Some constants
EPOCHS = 20
INITIAL_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
data_root_directory = './data/plantvillage/'
width = 256
heigth = 256
depth = 3
image_list, label_list = [], []

# convert images to numpy array using opencv
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None




# getting images from directory
def load_images():
    try:
        print("Loading images.....")
        root_dir = listdir(data_root_directory)
        for plant_folder in root_dir:
            plant_disease_folder_list = listdir(f"{data_root_directory}/{plant_folder}")

            for plant_disease_folder in plant_disease_folder_list:
                print(f"Processing {plant_disease_folder}")
                plant_disease_img_list = listdir(f"{data_root_directory}/{plant_folder}/{plant_disease_folder}")

                for image in plant_disease_img_list:
                    img = f"{data_root_directory}/{plant_folder}/{plant_disease_folder}/{image}"

                    if img.endswith(".jpg") or img.endswith(".JPG"):
                        image_list.append(convert_image_to_array(img))
                        label_list.append(plant_disease_folder)
        print("Image loading complete")
    except Exception as e:
        print(os.getcwd())
        print(f"Error: {e}")



def train_model():
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=43)
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest"
    )

    model = Sequential()
    input_shape = (heigth, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        input_shape = (depth, width, heigth)
        chanDim = 1
    
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(len(label_binarizer.classes_)))
    model.add(Activation("softmax"))

    print(f"Model Summary: \n{model.summary()}")

    opt = Adam(lr=INITIAL_LR, decay=INITIAL_LR / EPOCHS)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data = (x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS,verbose=1
    )

    scores = model.evaluate(x_test, y_test)

    print(f"Score: {scores}")

    return model


def save_model(model, file):
    pickle.dump(model, open(f"models/{file}.pkl", 'wb'))

def load_model(file):
    return pickle.load(open(f"models\{file}.pkl"), 'rb')