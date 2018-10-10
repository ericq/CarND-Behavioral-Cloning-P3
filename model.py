import csv
import cv2
import numpy as np
import os.path


# Setup Keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


## 
DATA_DIR_LIST=['../behavior-clone-training-data/1/',
        '../behavior-clone-training-data/2/',
        '../behavior-clone-training-data/3/',
        '../behavior-clone-training-data/4/',
        '../behavior-clone-training-data/5/',
        '../behavior-clone-training-data/6/',
        '../behavior-clone-training-data/7/']

# use left-cam as training img, 
#steering =  add this value to the current steer 
LEFT_CAM_ADJ = 0.2   

# use right-cam as training img, 
#steering =  add this value to the current steer 
RIGHT_CAM_ADJ = -0.2  

images = []
measurements=[]

## Load data. support multiple input data dir

for DATA_DIR in DATA_DIR_LIST:
    assert os.path.isdir(DATA_DIR), "data folder exists"

    lines = []
    driving_log = DATA_DIR+'driving_log.csv'
    assert os.path.isfile(driving_log), "driver log exists"

    with open (driving_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        source_path_leftcam = line[1]
        source_path_rightcam = line[2]
        #filename = source_path.split('/')[-1] #BUG here, does not work for windows system
        filename = os.path.basename(source_path)
        filename_leftcam = os.path.basename(source_path_leftcam)
        filename_rightcam = os.path.basename(source_path_rightcam)

        current_path = DATA_DIR+'IMG/' + filename
        current_path_leftcam = DATA_DIR+'IMG/' + filename_leftcam
        current_path_rightcam = DATA_DIR+'IMG/' + filename_rightcam

        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

        image_leftcam = cv2.imread(current_path_leftcam)
        images.append(image_leftcam)
        measurements.append(measurement+LEFT_CAM_ADJ)

        image_rightcam = cv2.imread(current_path_rightcam)
        images.append(image_rightcam)
        measurements.append(measurement+RIGHT_CAM_ADJ)


#Augment the images
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
assert len(X_train.shape) == 4, "X_train shape has 4 elements"
print(X_train.shape)

y_train = np.array(augmented_measurements)
print(y_train.shape)
assert len(y_train.shape) == 1, "y_train shape has 1 element"


#Augment the images
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

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

    ## Pre-process 
    # normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=input_shape))
    # cropping
    model.add(Cropping2D(cropping=((70,25),(0,0))))

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


batch_size = 64 
nb_epoch = 2

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


    
    
