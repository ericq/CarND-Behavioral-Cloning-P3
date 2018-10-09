import csv
import cv2
import numpy as np


# Setup Keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


DATA_DIR='./data/1/'

lines = []
with open (DATA_DIR+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements=[]


for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = DATA_DIR+'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)


X_train = np.array(images)
y_train = np.array(measurements)


def car_lenet(img_rows, img_cols, img_channels, dropout_keep_prob ):
    '''
    Creates the lenet

    Args:
        img_rows: number of rows (height)
        img_cols: number of columns (width)
        img_channels: number of channels
    	dropout_keep_prob: float, the fraction to keep before final layer.

    Returns:
    	logits: the logits outputs of the model.
    '''
    if K.image_data_format() == 'channels_first':
        input_shape = (img_channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols,img_channels)


    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))

    activation="relu"

    # define the first set of CONV => ACTIVATION => POOL layers
    model.add(Conv2D(20, 5, padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    # define the second set of CONV => ACTIVATION => POOL layers
    model.add(Conv2D(50, 5, padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    
    # define the first FC => ACTIVATION layers
    model.add(Flatten())

    #model.add(Dense(500))
    model.add(Dense(120))
    #model.add(Activation(activation))

    # define the second FC layer
    #model.add(Dense(numClasses))
    model.add(Dense(84))

    # lastly, define the soft-max classifier
    #model.add(Activation("softmax"))
    model.add(Dense(1))

    return model


def create_model(img_rows, img_cols, img_channels=3, dropout_prob=0.2):
    return car_lenet(img_rows, img_cols, img_channels, dropout_prob)


img_rows = 160 # resolution of inputs
img_cols = 320 # Resolution of inputs
img_channels = 3


batch_size = 16
nb_epoch = 10

model = create_model(img_rows, img_cols, img_channels )

# Learning rate is changed to 0.001
model.compile(optimizer='adam', loss='mse')

# Start Fine-tuning
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_split=0.2,
          shuffle=True
         )


model.save('model.h5')


    
    
