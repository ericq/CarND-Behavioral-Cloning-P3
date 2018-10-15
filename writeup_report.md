# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* run.mp4 containing the video of one full lap run
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (line 134 of model.py) consists of the following:
1. pre-processing set:
..* a normalization layer using Keras lambda to translate image RBG value to 0 and 1.0 with a mean value of 0.0. (line 146)
..* Further process the images by cropping top and bottom parts that are not helping the training. (line 148)
2. first set of convolution layer with the following:
..* a convolution neural network with 20 filter value and kernel size of 5 (model.py lines 153); 
..* followed by a layers to introduce nonlinearity (code line 154)
..* A maxPoling layer with pool size 2x2 and strides 2x2
3. second set of convolution layer by repeating the first set
4. add three full connected dense layers. (line 168, 173, 177)


#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

I have also experimented with adding dropout layer to the model. Add them between convlutional layers or between Dense layers. Also trired different drop prob ratio from 0.2 to 0.5. However, in all testing, the overall loss (~0.06/0.09) is worsened than the model without droput (~0.04). The simulation result was also worse with the dropout layer. So I removed the dropout layer in the end.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 203).

#### 4. Appropriate training data
I stored the training data in this github repository: https://github.com/ericq/behavior-clone-training-data

Training data was chosen to keep the vehicle driving on the road. I have used multiple laps of training data: 
lap1 - trained from udacity workspace, slow response to key controls but ok with low speed
lap 2 - trained from local simulator (windows env), pay attention to the image path in windows format generated in the csv file. higher speed
lap 3 - no steering as much as possible, then sharp correction before hit the edge
lap 4 - run the bridge and the last half lap 3 times
lap 5 - teach how to drive back to the center.  Still weaving back and forth between the middle of the road and the shoulder, but you need to turn off data recording when you weave out to the side, and turn it back on when you steer back to the middle.
lap 6 - post-bridge track back-n-forth
lap 7 - smooth drive in the center


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep it simple. Although the project instruction has provided reference to some more complicated models, like Nvidia CNN model and also I briefly tried the inception model during the work, I eventually chosed a simple model based on LeNet. I'm more interested to see what a simple model architecture is capable of. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. It's very important to use all 3 cameras data, and I found its vital to use proper correction offset for left camera and right cameras. After many experiment, I used +/- 0.35. 

To combat the overfitting, I keep on tuning the number of epochs and batch size. 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I focused more adding high quality data to the training set, rather than tuning the model too much itself. See the 7 sets of data I colleced above.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is almost identical to a normal LeNet.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used different strategies described above. 

After the collection process, I had 7 set (about 10 laps) of data points. I modified the basic input example so it can handel mulitple set of data input. This is more efficient than just put everything under one folder. See the line 23 where I specified which data set to use for the final model. 

Since I noticed that the data are very much concentrated on the steering angle as 0, so I tried various strategy to only keep a subset of those data. I noticed that the overall loss value (~0.19) is much higher than the one based before re-sampling (0.04) and also the simulation did not do well. It does not mean that the resmapling is not good, If I have more time, I'd like to collect many more laps of training data to continue the direction. In the final code I submitted, I did not include the re-sampling. The code with re-sampling is also included in the code model.with-resampling.py. 

Line 193 shows how to use sklearn train_test_split() function.




