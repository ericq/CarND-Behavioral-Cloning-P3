import csv
import cv2
import numpy as np
import os.path

import sklearn
from sklearn.model_selection import train_test_split

# Setup Keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


## Input data are stored under sub-directories. 
# so concatenate sub-directory into one dataset
#
ROOT_DATA_DIR = '../behavior-clone-training-data/'

#sub-set data dir want to process
SUBSET_DATA_DIR_LIST=['1/','2/','3/','4/','5/','6/','7/'] 

# use left-cam as training img, 
#steering =  add this value to the current steer 
LEFT_CAM_ADJ = 0.5   

# use right-cam as training img, 
#steering =  add this value to the current steer 
RIGHT_CAM_ADJ = -0.5  

# convert file names in the driving log csv to new file name
# this is because the simulator files might be copied to 
# different nodes. <root data dir>/<sub>/IMG/<img files>
# 
# input args:
#    fn_in_log: file name in log
#    new_root: new root directories
def getNewImgFn(fn_in_log, new_root):
    (pdir, fname) = os.path.split(fn_in_log)
    (pdir, up1) = os.path.split(pdir)
    (_,up2) = os.path.split(pdir)

    return os.path.join( ROOT_DATA_DIR,up2,up1,fname )
# Concatenate driving.log from all sub-direcotries into one
# lines will contain all driving-log
driving_logs = [] # contains tuple (img_name, angle)

for subdir in SUBSET_DATA_DIR_LIST:
    datadir = ROOT_DATA_DIR + subdir
    assert os.path.isdir(datadir), "data folder exists"

    driving_log = datadir +'driving_log.csv'
    assert os.path.isfile(driving_log), "driver log exists"

    subt = 0
    with open (driving_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # center camera
            driving_logs.append( (getNewImgFn(line[0],ROOT_DATA_DIR),
                    float(line[3])) )
            # left camera
            driving_logs.append( (getNewImgFn(line[1],ROOT_DATA_DIR),
                    float(line[3]) + LEFT_CAM_ADJ) )
            # right camera
            driving_logs.append( (getNewImgFn(line[2],ROOT_DATA_DIR),
                    float(line[3]) + RIGHT_CAM_ADJ) )
            subt += 1
    print("Read " + str(subt)+ " records from " + driving_log)

print("read total " + str(len(driving_logs)) +" driving records");

# trim off those records that always drive forward. 

# print a histogram to see which steering angle ranges are most overrepresented
# samples are arry of tuple [ (x1,x2,,,, y) ]
# this one will do full resampling based on y value distribution so it will be 
# more flatterened
# input args:
#     samples - input array of tuples
#    num_bins - a parameter can be tuned 
#    t_idx    - target value index of the tuple. default value is -1 (the last element)

def resampleByFlatterning( samples, num_bins=23, t_idx=-1):
    num_samples = len(samples)
    avg_samples_per_bin = num_samples/num_bins
    values = []
    for s in samples:
        values.append(s[t_idx])
    np_values = np.array(values)

    hist, bins = np.histogram(np_values, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)
    #plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    #plt.show()


    # determine keep probability for each bin: if below avg_samples_per_bin, keep all; 
    # otherwise keep prob is proportional to number of samples above the average, so as 
    # to bring the number of samples for that bin down to the average
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))
    remove_list = []
    for i in range(num_samples):
        for j in range(num_bins):
            if samples[i][t_idx] > bins[j] and samples[i][t_idx] <= bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
    num_tbd = len(remove_list) 
    for k in range(num_tbd):
        del samples[remove_list[num_tbd - k - 1]]

print("before resampling, driving logs entry number:"+str(len(driving_logs)))
resampleByFlatterning(driving_logs)
print("after resampling, driving logs entry number:"+str(len(driving_logs)))

# Generator to deal with large amount of data
# input args:
#   sampes - collection of driving record enntries
#   batch_log_size- size of those logs will be processed as one batch
#                   note that many(6) records generated from one log
def generator(samples, batch_log_size=32):
    print("entering...")
    num_samples = len(samples)
    print("received "+str(num_samples)+" records")

    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_log_size):
            batch_samples = samples[offset:offset+batch_log_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                # Read center camera; also add a flipped one
                ic = cv2.imread(batch_sample[0])
                ac = batch_sample[1]

                images.append(ic) 
                angles.append(ac)
                images.append(cv2.flip(ic,1))
                angles.append( ac*-1.0 ) 

            # trim image to only see section with road
            X_train = np.array(images)
            assert len(X_train.shape) == 4, "X_train shape has 4 elements"
            y_train = np.array(angles)
            assert len(y_train.shape) == 1, "y_train shape has 1 element"
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield ( X_train, y_train)


##
#    Creates the lenet

#    Args:
#        img_rows: number of rows (height)
#        img_cols: number of columns (width)
#        img_channels: number of channels
#    	dropout_keep_prob: float, the fraction to keep before final layer.
#
#    Returns:
#    	logits: the logits outputs of the model.


def car_lenet(img_rows, img_cols, img_channels, dropout_keep_prob ):
    
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

batch_log_size = 32 # 2X
nb_epoch = 2

train_samples, validation_samples = train_test_split(driving_logs, 
                                                    test_size=0.2)

# NOTE: acutal number fed into training is 6 * batch_size
train_generator = generator(train_samples, batch_log_size=batch_log_size)
validation_generator = generator(validation_samples, batch_log_size=batch_log_size)

model = create_model(img_rows, img_cols, img_channels )

# Learning rate is changed to 0.001
model.compile(optimizer='adam', loss='mse')

# Start Fine-tuning
# steps_per_epoch: Integer. Total number of steps (batches of samples) to yield 
# from generator before declaring one epoch finished and starting the next epoch. 
# It should typically be equal to the number of samples of your dataset divided 
# by the batch size
model.fit_generator(train_generator,
                    steps_per_epoch = len(train_samples) / batch_log_size,
                    validation_data=validation_generator, 
                    validation_steps= len(validation_samples) / batch_log_size,
                    epochs=nb_epoch)

model.save('model.h5')


