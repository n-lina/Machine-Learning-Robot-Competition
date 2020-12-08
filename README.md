# Machine Learning Robot Competition
## Competition Overview

**Task:** To atonomously navigate the simulated world via a live-image feed, avoiding moving obstacles like pedestrians and the truck, and to accuractely recognize and read alphanumeric "license plates" on parked cars in the simulation using machine learning principles. 

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
    - Since our only data input was live image feed from the simulation, we decided to use the unique color that elements of interest had i.e. the blue colour of the boxes, the white lines. Hence, for most of our decision making we utilize colour masking after converting our RGB image feed to HSV. 
- Convolutional Neural Network: a neutral network with three layers, trained and validated using input data we collected and provided. 

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
-  Robot is too far from the line: TURN RIGHT
-  Robot is too close to the line: TURN LEFT
-  Drive Straight otherwise <br>
  <br>

The Proportional component was calculated by multiplying by a constant the difference between the robot's last position and current position. 

The Derivative component was calculated by multiplying by a constant the change in the robot's position over time. 

## Neural Network For Alphanumeric Character Detection 

### License Plate Detection 
Each box contained two rectangular contours. The first one had the information for
the location and the second one consisted the license plate. Due to the dimensions
of the first contour for the location we realized that it was more reliable detecting it
while navigating, hence our algorithm followed the following approach.
-  Continuously monitor the latest frame for a location plate after masking it
with respect to its colour
-  Use the aspect ratio between the location plate and license plate which was
experimentally determined, to approximate the position of the license plate
-  Mask the license plate with a blue filter and convert it into binary
-  Retrieve a rectangle contour for each letter and number
-  Pass each contour through a CNN to retrieve each value <br>
<br>
As a factor of safety and to reduce the computational complexity of our system
we only passed through the CNN the latest set of letter/number contours that we
retrieved, after crossing a parallel line that preceded a box. This enabled us to take the most accurate reading from the batch after our robot had the most time to
correct its position. This also reduced the necessity of a high performance CNN,
since our input data was of low variation.

### Convoluted Neural Network 
The architecture of our Convolutional Neural Network is as Following:

``` py 
from k e r a s impor t m o del s
conv_model = m o del s . S e q u e n t i a l ( )
#BLOCK 1
conv_model . add ( l a y e r s . Conv2D ( 1 6 , ( 3 , 3 ) , a c t i v a t i o n = ’ r e l u ’ ,
i n p u t _ s h a p e = ( 3 2 , 3 2 , 3 ) ) )
conv_model . add ( l a y e r s . MaxPooling2D ( ( 2 , 2 ) ) )
#BLOCK 2
conv_model . add ( l a y e r s . Conv2D ( 1 6 , ( 3 , 3 ) , a c t i v a t i o n = ’ r e l u ’ ) )
conv_model . add ( l a y e r s . MaxPooling2D ( ( 2 , 2 ) ) )
#BLOCK 3
conv_model . add ( l a y e r s . Conv2D ( 3 2 , ( 3 , 3 ) , a c t i v a t i o n = ’ r e l u ’ ) )
conv_model . add ( l a y e r s . MaxPooling2D ( ( 2 , 2 ) ) )
conv_model . add ( l a y e r s . F l a t t e n ( ) )
conv_model . add ( l a y e r s . D r o p o ut ( 0 . 3 ) )
conv_model . add ( l a y e r s . Dense ( 2 5 6 , a c t i v a t i o n = ’ r e l u ’ ) )
conv_model . add ( l a y e r s . Dense ( 3 6 , a c t i v a t i o n = ’ s o ftm a x ’ ) )
```

A summary of the model can be found in appendix A Figure A.2.

While designing the architecture of the CNN we noticed that the average
shape of the letter/number contours that we extracted was around 28x30 pixels
hence to avoid the distortion of information while interpolating when resizing we
chose an input shape of 32x32 pixels, and hence the maximum amount of layers
(Convolution-Max Pooling) our model could be was three.

### Convoluted Neural Network Training and Validation 
As mentioned in the previous section, our approach for finding the letters/numbers
of the license plate reduced the dependency of high performance CNN, and hence
a were able to generate custom dataset from the original blank plate. As a factor
of safety though we implemented certain measurements to ensure that we are
diversifying and not over-fitting our data. We first used Gaussian Blur to simulate
the lower image quality of the expected data and then generated a diverse set of
images using the ImageDataGenerator class from Keras as shown below:

```py
datagen = ImageDataGenerator(
      rescale = 1/255,
      shear_range = 25.0,
      brightness_range = [0.2, 1.0],
      zoom_ range = [0.5, 1.5],
      h o r i z o n t a l _ f l i p = F al s e , )
h i s t o r y _ c o n v = conv_model . f i t _ g e n e r a t o r ( d at a g e n . fl ow ( X _t r ai n ,
Y _t r ai n , b a t c h _ s i z e = 3 2 ) , v a l i d a t i o n _ d a t a = ( X _t e st , Y _ t e s t ) ,
s t e p s _ p e r _ e p o c h = l e n (X) / 3 2 , e p o c h s = 6 0 )
```
We generated 3000 images which we passed through our DataGenerator with
validation split of 0.2. Model loss and model accuracy plots can be found in the
Appendix

### Plate Detection Conclusion 
Reflecting on our plate detection model, it performed as expected during the
competition, although it missed one of the outer plates, which was an issue that
we had observed through testing. It was an issue that we were able to significantly
reduce its occurrence but not completely eliminate. The difficulty of fixing this
problem, is that there are many factors that could cause such error i.e. the HSV
masking thresholds, the constrictions on the expected size of the contour. In future
iterations we would revisit each part individually and try to optimize, as well as
exploring new approaches to cross validate results.

## Object Detection 
### Pedestrian Detection 
In order to detect the pedestrians we utilized the distinct colour of their lower
body by converting the frame to HSV and masking it based on the thresholds that
we experimentally determined. To reduce redundant computational expenditure
we only monitored for pedestrians once the position of the robot reached on a
parallel line positioned before the crosswalks. In order to ensure that no collision
would occur our agent would wait first for the pedestrian to pass once through the
crosswalk and then continue navigating
### Vehicle Detection 
Similar to the pedestrian detection we utilized distinct colours from the vehicle to
mask our frame, the thresholds were experimentally determined. To ensure that
we don’t collide with the vehicle and to avoid having to monitor its position while
we are navigating the inner loop, before entering the inner loop we would wait until the position of the vehicle is far enough such that we would safely not get in
contact with the vehicle
### Conclusion 
The Object Detection was one of the last modules we implemented, which is why
we were time constricted when optimizing it. This is why we emphasized on
developing a safe system rather than a fast one,which of course came as a trade-off
in run time. We were satisfied in using HSV masking to detect both the vehicle
and pedestrians, but in future iterations we would revisit our strategy for handing
both pedestrian and vehicle detection.
Some improvements for the pedestrians could be monitoring the red line before the
crosswalk instead the robots outer loop position, since the red line is closer to the
cross walk and would allow a faster response. In regards of the vehicle handling we
would explore the approach of constantly monitoring the vehicle, while entering
directly to the inner loop. This is an approach that we did not have sufficient time
to test and compare with our metod.





