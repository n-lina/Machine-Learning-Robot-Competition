# Machine Learning Robot Competition
## Competition Overview

**Task:** To atonomously navigate the simulated world via a live-image feed, avoiding moving obstacles like pedestrians and the truck, and to accuractely parse alphanumeric "license plates" on parked cars in the simulation using machine learning principles. 

**Competition Criterion:** Points are awarded for every license plate the convoluted neural network accurately parses, and in the case of a tie, by time robot took to complete the competition course.

**Result:** Placed 4th out of 20 teams!

<br>
<pre>Competition Surface Rendered in Gazebo</pre>
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/compSurface.PNG?raw=true" width="600" height="600"/>
<br>

### Technologies Used 
-   **Gazebo Physics Engine:** the 3D Robotics simulation that served as the competition surface 
-   **Robot Operating System:** a flexible framework for writing robot software, used to define the robot object
-   **Python Programming Language:** used to process the robot's interactions with its environment; used to implemenent autonomous navigation, alphanumeric character detection via a neural network, and object detection via OpenCV
    -    **OpenCV:** used Open Computer Vision to allow the robot to sense its environment and to avoid pedestrians and the truck. 
-   **Keras:** a deep learning API written in Python, running on top of Tensorflow
-   **Tensorflow:** the open-source library for a number of various tasks in machine learning
    -   Keras and Tensorflow were used to develop, train, and test the convoluted neural network responsible for alphanumeric license plate parsing. 
    

## Software Components 
- main python script: contains image processing functions and control algorithms. 
- Robot class: responsible for interactions with the Gazebo simulation. 
- Convolutional Neural Network: a neutral network with three layers, trained and validated using input data we generated. 

## Neural Network For Alphanumeric Character Detection 

### License Plate Detection 
Using **colour masking** and looking for the known aspect ratio of the license plate, we extracted the license plate from the robot's live-image feed. While we discuss ideal images of license plates next, the license plates extracted from the Gazebo world were often sheared, blurry, and imperfect due to the robot's motion and angle! 

Using a python script, we generated thousands of license plates, extracted their characters, and input them into our neural network for training.

<pre> Example License Plate </pre> 
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/plate.png?raw=true" width = "200"/>
<pre> Data generation and Neural Network Training Pipeline </pre>  
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/cnnPipeline.PNG?raw=true" width="500"/>


### Convoluted Neural Network 
The architecture of our Convolutional Neural Network is as Following:

``` py 
from keras import models
conv_model = models.Sequential()
#BLOCK 1
conv_model.add(layers.Conv2D(16,(3,3),activation=’relu’, input_shape = (32, 32, 3)))
conv_model.add(layers.MaxPooling2D((2, 2)))
#BLOCK 2
conv_model.add(layers.Conv2D(16,(3,3),activation = ’relu’))
conv_model.add(layers.MaxPooling2D((2,2)))
#BLOCK 3
conv_model.add(layers.Conv2D(32,(3,3),activation= ’relu’))
conv_model.add(layers.MaxPooling2D (( 2 , 2 )))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.3))
conv_model.add(layers.Dense(256,activation = ’relu’))
conv_model.add(layers.Dense(36,activation = ’softmax’))
```

<pre> Summary of the CNN model (DELETE: appendix A Figure A.2.) </pre> 
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/cnnModelSummary.PNG?raw=true" width="400"/>

Designing the architecture of the CNN, we noticed that the average character extracted from a perfect license plate was 28x30 pixels, but around 32x32 piexels from the Gazebo world. To avoid inaccuracy due to distortion, we trained our CNN using 32x32 pixel images instead, making our "Convolution-Max Pooling" value three. 

### Convoluted Neural Network Training and Validation 
To best replicate the actual inputs coming from the Gazebo simulation, we used Gaussian Blur to lower the image quality of the perfect license plates. We also targeted 'difficult' characters, like 'B' vs. '8' or '1' vs. 'I', by generating an abundance of input data with these characters. We used **Keras** ImageDataGenerator to then generate a representative collection of inputs: 

```py
datagen = ImageDataGenerator(
      rescale = 1/255,
      shear_range = 25.0,
      brightness_range = [0.2, 1.0],
      zoom_ range = [0.5, 1.5],
      horizontal_flip = False)
history_conv = conv_model.fit_generator(datagen.flow(X_train, Y_train, batch_size= 32), validation_data = (X_test, Y_test),
steps_per_epoch = len(X)/32 , epochs = 60)
```
We generated 3000 images which we passed through our DataGenerator with a validation split of 0.2. Pictured below are our model loss and model accuracy plots:  
<pre> Model Loss Plot  </pre> 
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/cnnModelLoss.PNG?raw=true" width="300"/>
<pre> Model Accuracy Plot </pre> 
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/cnnModelAccuracy.PNG?raw=true" width="300"/>

## Robot Class
The robot class is responsible for interactions with the simulated world. 

### Constructor 
The robot object is initialized with: 
-  an inner loop position tracker for navigating the inner loop
-  an outer loop position tracker for navigating the outer loop 
-  an image view attribute to contain the latest image from coming from the live-image feed. 
-  a ROS Subscriber to the image topic from Gazebo 
-  a ROS Publisher to report the license plate characters that the neural network parses 
-  a ROS Publisher to change the linear and angular velocity of the robot for nagivation purposes. 

### Methods 
-   __getImage:
    -   Subscriber callback function
    -   converts the Gazebo 'Image' into an OpenCV RGB image using the 'cv_bridge' python package 
    -   updates the robot's 'view' attribute 
-   publishLicensePlate: 
    -   publishes the parsed license plate using the ROS Publisher attribute of the robot 
-   linearChange, angularChange: 
    -   controls the navigation of the robot 
- imageSliceVer, imageSliceHor, imageSliceVertical: 
    -   using the 'cv2' python package, these methods process the live-feed image and aid in navigation. 

## Autonomous Navigation 
### Position Tracking
Accurate position tracking is imperative for autonomous navigation. Paying very close **attention to detail**, we noticed that the competition track has markings that are slightly lighter than the rest of the track. We decided to use these lines to track the robot's position on the track. Fun fact: we were the only partnership to notice this detail and it ended up being very helpful! 

Using **colour masking** in OpenCV, we produced a mask in which the grey lines were bright white while the rest of the image was completely black. This allowed the robot to easily detect the white lines. 

In order to avoid double-counting the marker lines, we also added a delay between lines as a "debouncer." 

<br>
<pre>We tracked the robot's position by noting the number of light-grey lines passed ... </pre>
<img src="https://github.com/n-lina/Machine-Learning-Robot-Competition/blob/master/position.png?raw=true" width="600" height="600"/> 
<br>

### Proportional-Derivative Navigation Algorithm 
In order to navigate across the outer loop, we monitored the robot's distance to outer perimeter of the road and adjusted its navigation accordingly via a **Proportional-Derivative control algorithm**. 
-  Robot too far from the perimeter: TURN RIGHT
-  Robot is too close to the perimeter: TURN LEFT
-  Drive straight otherwise <br>
The Proportional component was calculated by multiplying by a constant the difference between the robot's last position and current position. 

The Derivative component was calculated by multiplying by a constant the change in the robot's position over time. 

## Object Detection 
### Pedestrian and Truck Detection 
Once again, we used **HSV colour masking** in **OpenCV** to detect and avoid pedestrians and the truck. 






