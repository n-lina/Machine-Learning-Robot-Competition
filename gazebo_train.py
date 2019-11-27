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

#import qlearn
#import liveplot
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
from sensor_msgs.msg import Image

from matplotlib import pyplot as plt
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import string
import robot
#import ros_methods as rm


def get_plate(model, letters, invert_dict):
    plate = []
    for pic in letters:
        size = (32, 32)
        img = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        #img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = np.array(img)/255.0
        img_aug = np.expand_dims(img, axis=0)
        y_predict = conv_model.predict(img_aug)[0]
        result_int = np.argmax(y_predict)
        result = invert_dict[result_int]
        plate.append(result)
    license_plate = string.join(plate)
    print(license_plate)
    return


def get_encoder(data):
    data = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    encoded = to_categorical(integer_encoded)
    invert_dict = dict(zip(integer_encoded, data))
    return encoded, invert_dict


def filter_blue(original_img):
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    low = np.uint8([[[80, 0, 0]]])
    up = np.uint8([[[200, 96, 96]]])

    hsv_low = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    hsv_high = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    #print("LOW: {}\nHIGH: {}".format(hsv_low, hsv_high))

    # Threshold of blue in HSV space
    lower_blue = np.array([hsv_low[0][0][0], hsv_high[0][0][1], hsv_low[0][0][2]])
    upper_blue = np.array([hsv_high[0][0][0], hsv_low[0][0][1], hsv_high[0][0][2]])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def find_letters(binaryImg, drawOn):
    contours, _ = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnt = []
    dimensions = []
    #n = 0
    #cwd = os.curdir
    for cnt in contours:
        # area = cv2.contourArea(cnt)
        # if area > 25: hsv_high[0][0][0]
        #     filtered_cnt.append(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # if (w < 15 or h < 14 or h > 25 or w > 25):
        #     continue
        #copy_img = drawOn
        #filtered_cnt.append(cnt)
        if x-3 < 0:
            x = 0
        else:
            x = x-3
        corners = (x, y-2, w+6, h+6)
        dimensions.append(corners)
        #drawOn = cv2.rectangle(drawOn, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow("draw", drawOn)
    # cv2.waitKey(3)
    dimensions = sorted(dimensions, key=lambda l: l[0])


    if len(dimensions) == 4:
        for dim in dimensions:
            cropped_img = drawOn[dim[1]:dim[1]+dim[3], dim[0]:dim[0]+dim[2]]
            filtered_cnt.append(cropped_img)
            #copy_img = cv2.rectangle(copy_img, (dim[0], dim[1]), (dim[0]+dim[2], dim[1]+dim[3]), (0, 255, 0), 2)
            # pts1 = np.float32([[dim[0], dim[1]], [dim[0]+dim[2], dim[1]], [dim[0], dim[1]+dim[3]], [dim[0]+dim[2], dim[1]+dim[3]]])
            # pts2 = np.float32([[0, 0], [32, 0], [0, 32], [32, 32]])
            # matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # cropped_img = cv2.warpPerspective(drawOn, matrix, (32, 32))
            #num = random.randint(0, 1000)
            #cv2.imwrite("{}/croped{}.png".format(cwd, x), cropped_img)
            #filtered_cnt.append(cropped_img)
            #n += 1
        return True, filtered_cnt
        #cv2.imshow("result", copy_img)
        #cv2.waitKey(3)
    #markedImg = cv2.drawContours(dr, contours, -1, (0, 0, 255), 3)
    return False, filtered_cnt


def find_contours(binaryImg, quarter_drawOn):
    contours, _ = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cwd = os.curdir
    plates_array = []
    for cnt in contours:
        #print("contour area: {}".format(cv2.contourArea(cnt)))
        x, y, w, h = cv2.boundingRect(cnt)
        #print("bounding rect area: {}".format(w*h))
        area_ratio = float(cv2.contourArea(cnt)/(w*h))
        #aspect_ratio = float(w/h)
        #or area_ratio < 0.9
        #aspect_ratio < 0.95 or aspect_ratio > 1.2 or
        if area_ratio < 0.9 or h < 125 or w < 130: 
            continue
        #location = quarter_drawOn[y+(h/2):y+h, x:x+w]
        #print("height: {}".format(h))
       # offset = int(0.3*h)
        plate = quarter_drawOn[y+h:y+h+35, x:x+w] #adjust this 
        # cv2.imshow("crop", location)
        # cv2.waitKey(3)
        plates_array.append(plate)
       # quarter_drawOn = cv2.rectangle(quarter_drawOn, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #img_1 = cv2.drawContours(drawOn, contours, -1, (0,255,0), 3)
    #cv2.imshow("draw", quarter_drawOn)
    #cv2.imshow("countor", img_1)
    #cv2.waitKey(3)
    if len(plates_array) != 0: 
        return True, plates_array
    return False, plates_array


def contains_white_line(cropped_original_img):
    hsv = cv2.cvtColor(cropped_original_img, cv2.COLOR_BGR2HSV)
    grey_line = np.uint8([[[95, 96, 95]]])
    green_line = np.uint8([[[118, 128, 111]]])
    hsv_low = cv2.cvtColor(grey_line, cv2.COLOR_BGR2HSV)
    hsv_high = cv2.cvtColor(green_line, cv2.COLOR_BGR2HSV)
    #print("LOW: {}\nHIGH: {}".format(hsv_low, hsv_high))
    grey = np.array([hsv_low[0][0][0], hsv_low[0][0][1], hsv_low[0][0][2]])
    green = np.array([hsv_high[0][0][0], hsv_high[0][0][1], hsv_high[0][0][2]])
    mask = cv2.inRange(hsv, grey, green)
    cv2.imshow("mask", mask)
    cv2.waitKey(3)
    return cv2.inRange(mask, 255, 255).any()

def contains_human(cropped_original_img):
    hsv = cv2.cvtColor(cropped_original_img, cv2.COLOR_BGR2HSV)
    lower = np.uint8([[[41, 20, 14]]])
    upper = np.uint8([[[158, 134, 107]]])
    hsv_low = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    hsv_high = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    #print("LOW: {}\nHIGH: {}".format(hsv_low, hsv_high))
    lower_limit = np.array([hsv_high[0][0][0], hsv_high[0][0][1], hsv_low[0][0][2]])
    upper_limit = np.array([hsv_low[0][0][0], hsv_low[0][0][1], hsv_high[0][0][2]])
    mask = cv2.inRange(hsv, lower_limit, upper_limit)
    #cv2.imshow("mask", cropped_original_img)
    #cv2.imshow("mask1", mask)
    #cv2.waitKey(3)
    #print("hello")
    return cv2.inRange(mask, 255, 255).any()


def filter_plate(quarter_original_img):
    hsv = cv2.cvtColor(quarter_original_img, cv2.COLOR_BGR2HSV)
    lower_limit = np.uint8([[[99, 100, 99]]])
    upper_limit = np.uint8([[[203, 203, 203]]])
    hsv_low = cv2.cvtColor(lower_limit, cv2.COLOR_BGR2HSV)
    hsv_high = cv2.cvtColor(upper_limit, cv2.COLOR_BGR2HSV)
    #print("LOW: {}\nHIGH: {}".format(hsv_low, hsv_high))
    lower = np.array([hsv_high[0][0][0], hsv_high[0][0][1], hsv_low[0][0][2]])
    upper = np.array([hsv_low[0][0][0], hsv_low[0][0][1], hsv_high[0][0][2]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


K = 2
D = 20
timer = 1
STRAIGHT = 1
BACK = -1
STOP = 0
LEFT = 1
RIGHT = -1
WHITE = 255
labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
box_positions = {2: "1", 4: "2", 8: "3", 10: "4", 12: "5", 0: "6"}

if __name__ == '__main__':
    # env = gym.make('Gazebo_Train-v0')
    # myrobot = robot.Robot()
    # time.sleep(7)
    # while True:
    # #    cv2.imshow("view", myrobot.view)
    # #    cv2.waitKey(3)
    #    contains_human(myrobot.view[390:460, 530:750])

    env = gym.make('Gazebo_Train-v0')
    STATE = 0
    myrobot = robot.Robot()
    conv_model = models.load_model("/home/fizzer/enph353_gym-gazebo/examples/gazebo_train/text_cnn4.h5")
    _, invert_dict = get_encoder(labels)
    time.sleep(10)
    myrobot.linearChange(STRAIGHT)

    while STATE == 0:
        img_slice = myrobot.imageSliceVer()
        for i in range(720):
            if img_slice[i] == WHITE and i > 718: 
                myrobot.angularChange(LEFT)
                STATE = 1
                break
    previous_state = 1000
    got_letters = False

    pedestrian_passed = False
    black = 0 
    while STATE == 1:
        img_slice = myrobot.imageSliceHor()
        cropped_original = myrobot.view[650:, :]

        if(contains_white_line(cropped_original)):
            if black > 150:
                black = 0
                myrobot.position = (myrobot.position + 1)%16
                print(myrobot.position)
                pedestrian_passed = False
                new_line = False 
        else: 
            black += 1

        if(pedestrian_passed is False and myrobot.position == 5 or myrobot.position == 13):
            while(pedestrian_passed is False and not contains_human(myrobot.view[390:460, 530:750])):
                myrobot.linearChange(STOP)
            while(contains_human(myrobot.view[390:460, 530:750])):
                myrobot.linearChange(STOP)
            pedestrian_passed = True
    
        #if((myrobot.position + 1) in box_positions):
        quarter_original_img = myrobot.view[360:, :500]
        plate_mask = filter_plate(quarter_original_img)
        success, plates = find_contours(plate_mask, quarter_original_img)
        if success:
            masked_plate = filter_blue(plates[len(plates) - 1])
            got_letters, letters = find_letters(masked_plate, plates[len(plates) - 1])
        if got_letters and myrobot.position in box_positions:
            get_plate(conv_model, letters, invert_dict)
            print(box_positions[myrobot.position])
            got_letters = False

        for i in reversed(range(600, 1280)):
            if img_slice[i] == WHITE:
                break
        error = i - previous_state
        d_error = 0
        if np.abs(error):
            d_error = error/timer
            timer = 0
        diff = K*error
        derivative = D*d_error
        total_error = diff + derivative
        if i < 1140 - total_error:
            myrobot.angularChange(LEFT)
        elif i > 1260 - total_error:
            myrobot.angularChange(RIGHT)
        else:
            myrobot.linearChange(STRAIGHT)
        previous_state = i
        timer += 1
  

