# Machine Learning Robot Competition
## Competition Overview
The scope of this project is to develop a software agent that would control a
robot and navigate across a simulated world while reporting license plates and
their location. While navigating the agent should be able to avoid obstacles such
as other vehicles and pedestrians and follow standard traffic rules. The simulation
was built in Gazebo and the agent takes live image feed from the in-simulation
camera mounted on top of the robot. The agent must interact with the world in
ROS, while it is scripted using python 2.7. The competition itself consists of a four
minutes trial where the agent must individually and autonomously run along the
track and report as many license plate possible

## Project Overview 
In order to target this project we decided from the early stages our general software
architecture. We concluded with having a main script that would handle the image
processing and decision making of the agent, and a class that would handle the
interaction between the simulation and our main script. Since our only data input
was live image feed from the simulation, we decided to use the unique color that
elements of interest had i.e. the blue colour of the boxes, the white lines. Hence,
for most of our decision making we utilize colour masking after converting our
RGB image feed to HSV. In regards of the text recognition, we implemented a
Convolutional Neural Network with three layers which we trained and validated
through custom data that we generated.

## Task Breakdown 
In order to reach our goal we had to break down the task in several modules. The
full Project Breakdown can be found listed below. An overview for each module is
explored in its representative section
-  Robot Class
-  Navigation
-  License Plate Detection & Recognition
-  Obstacle Detection & Handling

## Robot Class
In this section we explore the class we created to handle the interaction between
the Gazebo world and our main script. We ensured to utilize public and private
attributes/methods, in order to separate, encapsulate our systems and highlight the
unique interaction of the Robot class with the Gazebo simulation.

### Constructor 
The attributes of the Robot object initialized consist an inner loop position tracker
and an outer loop position tracker of our robot which describe the position of our
robot in regards of the number of parallel lines we have encountered on the road
(more on this in the navigation section),and an image view attribute which holds
the latest image feed from the simulation.
We also also initialize three private objects, a Subscriber to the image topic from
the Gazebo simulation,a Publisher to report the license plate information after we
have successfully identified one and another Publisher to change the linear and
angular velocity of the robot. 

### Methods 
We use a collection of public and private methods in order to serve the needs of
the Robot Class, and perhaps the most important is the subscriber Callback. Its
operation works as follows, whenever a new image is published from the simulation
our callback function, converts the Gazebo Image to an OpenCV RGB image, by
using the bridge python package, then it updates our image view attribute, which is
then continuously used by our main algorithm.

## Autonomous Navigation 
### Position Tracking
One of the most crucial components of our system was our position tracking, since
this information was utilized for most of our decision making of our agent. As
one can see from Figure A.1 throughout the competition route there are parallel
white lines between sections of the road. That is something we noticed towards the
later stages of the competition, and decided to exploit, since it would allow us to
eliminate tasks such as recognizing the plate location, and it provided a smoother
and reliable transition when switching from the outer loop to the inner loop.
In order to identify the parallel lines, we collected data with different RGB
values of the road and then took the lower and upper thresholds of our interest and
applied masking to our frame after converting it to HSV. The challenging part of
this process was to take into account the different lighting, and hence different
RGB values that occurred across the map. Thankfully the contrast between the
darker section of the road and the parallel lines was significant enough, to return
us the desired output. Another challenge we faced with was to avoid recounting
a parallel line. This mainly occurred, when the bot had to change for a longer
period its angular velocity while correcting its position. In order to fix this bug,
we incorporated a debouncer in the system. To identify the optimal threshold, we
tested our algorithm with different real-time factors.

### Navigation: Outer Loop
In order to navigate across the outer loop, we constantly monitored the outer white
line.By knowing the distance of our robot from the line we were able to create a
three case scenario using a fixed threshold and by navigating CCW:
-  Robot is too far from the line: TURN RIGHT
-  Robot is too close to the line: TURN LEFT
-  Drive Straight otherwise <br>
  <br>
This system allowed us to safely navigate across the outer loop without braking
any of the traffic rules. We shortly noticed though, that due to the fixed thresholds
our navigation was very dependent towards the initial condition, and if we didn’t
start in an optimal position while entering the outer loop, our license plate readings
were heavily affected. Since our initial state was very unreliable we wanted to
create a resilient system that would correct itself, regardless of the initial conditions,
and hence we tried to introduce PID control. The greatest challenge with introducing PID control, was that both the angular
and linear velocities of the robot were fixed, but we shortly noticed that that we
could alter our fixed threshold states. We included a Proportional and Derivative
error which were calculated as following:
The Proportional Error by the difference of the distances between the previous
and current measurement multiplied by a constant
The Derivative Error, by the change of distance throughout time multiplied by a
constant. 

### Navigation: Inner Loop 
Similar to the outer loop we followed a similar approach for the inner loop, but
instead moving CW now and following the outer left line. The main difference
between inner and outer loop, was that the white line that covers the outside
perimeter of the inner loop is not continuous. We were able to solve this issue by
utilizing our reliable navigation system and position tracking. Although it can not
be clearly seen in the Figure A.1 similar to the outer loop there are parallel lines
in the inner loop, which we kept on track as well. There was always a parallel
line before and after a discontinuity, and hence when entering one we could just
drive straight till we reach the end of the discontinuity, our PID control then would
correct our position if we entered the discontinuity with an angle.

### Initial Condition 
Finally in regards of our initial condition, our algorithm consisted of driving straight
till it detected a white line towards the bottom of the frame, after that our regular
outer loop line following protocol would commence. A great challenge that we
faced in this project in regards of the initial condition, was that the in several test
runs, our system would identify the white line sooner than expected, which would
cause an error when entering our main algorithm. We were able to significantly
reduce the occurrence of this error, by constricting the threshold in which the line
could be identified, but we were unable to fully fix the bug.

### Navigation Conclusion 
We were satisfied with the overall performance of our navigation module, although
there are certainly improvements that we would consider for future iterations.
We would try to optimize our initial condition handling. Perhaps by taking a
greater portion of the frame and normalizing it to find the average distance from
our robot to the white line, which would potentially give more reliable readings.
We could also explore new methods such as initially following the inner left line,
and switching to the outer line once we have crossed a parallel line prior to the first
box.

## Neural Network For Alphanumeric Characters

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
d at a g e n = Im a g e D at a G e n e r at o r (
      r e s c a l e = 1 . / 2 5 5 ,
      s h e a r _ r a n g e = 2 5 . 0 ,
      b r i g h t n e s s _ r a n g e = [ 0 . 2 , 1 . 0 ] ,
      zoom_ range = [ 0 . 5 , 1 . 5 ] ,
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





