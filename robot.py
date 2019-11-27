import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time
import os
from keras import models
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


import cv2
import rospy
import roslaunch
import time
import numpy as np

import qlearn
import liveplot
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
from sensor_msgs.msg import Image

from matplotlib import pyplot as plt
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import ros_methods as rm
import string


class Robot:
    def __init__(self):
        self.inner_position = 0
        self.position = 0
        self.view = []
        rospy.Subscriber("R1/pi_camera/image_raw", Image, self.__getImage)
        self.__vel = rospy.Publisher('/R1/cmd_vel', Twist)
        self.__plate = rospy.Publisher('/license_plate', String)
    
    def __getImage(self, data):
        try:
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.view = cv_image
    
    def linearChange(self, speed):
        vel_msg = Twist()
        vel_msg.angular.z = 0
        vel_msg.linear.x = speed
        self.__vel.publish(vel_msg)

    def angularChange(self, speed):
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = speed
        self.__vel.publish(vel_msg)

    def imageSliceVer(self):
        img = self.view[:, 500:501]
        threshold = 128
        cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, cv_bin = cv2.threshold(cv_grey, threshold, 255, cv2.THRESH_BINARY)
        return cv_bin[:, 0]

    def imageSliceHor(self):
        threshold = 128
        img = self.view[700:701, :]
        cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, cv_bin = cv2.threshold(cv_grey, threshold, 255, cv2.THRESH_BINARY)
        return cv_bin[0, :]
