# coding=utf-8
import naoqi
import qi
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from naoqi import ALProxy
from naoqi import ALModule

"""
Class Robot represents a simple library for working with NAO robots.
"""
class Robot:
    def __init__(self, ip, port):
        # initialization of naoqi modules
        self.posture    = ALProxy("ALRobotPosture", ip, port)
        self.tts        = ALProxy('ALTextToSpeech', ip, port)
        self.motion     = ALProxy("ALMotion",       ip, port)
        self.mood       = ALProxy("ALMood",         ip, port)
        self.proxy      = ALProxy("ALVideoDevice",  ip, port)
        
        self.tts.setLanguage('English')
        self.tts.setVolume(0.6)

    def get_image(self):
        # Initialization and connection to robot's camera.
        self.proxy.unsubscribeAllInstances("cam")
        name        = 'cam'
        cam_id      = 0
        resolution  = 2  # 640*480
        color_space = 13 # format BGR
        fps         = 30
        cam         = self.proxy.subscribeCamera(name, cam_id, resolution, color_space, fps)
        
        image       = self.proxy.getImageRemote(cam)
        im          = image[6] # creation of byte array
        nparr       = np.fromstring(im, np.uint8) # converting it to numbers
        nparr = nparr.reshape(480, 640, 3)
        
        #cv2.imshow("camera", nparr) # show result image
        #self.proxy.unsubscribeCamera(cam) #unsubscribe from robot's camera
        #cv2.waitKey(0) #close window with result image after pressing any key

        return nparr              

    # postures:
    def stand(self):
        self.posture.goToPosture("Stand", 0.5)

    def sit(self):
        self.posture.goToPosture("Sit", 0.5)

    def make_posture(self, posture):
        self.posture.goToPosture(posture, 0.5)

    # movement:
    def move_to_coords(self, x, y, z):
        self.motion.moveTo(x, y, z, 0.6)

    # speaking
    def set_czech(self):
        self.tts.setLanguage('Czech')

    def set_english(self):
        self.tts.setLanguage('English')

    def say_text(self, text):
        self.tts.say(text)


def params_setter():
    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 20000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    return params


class BallDetection:
    def __init__(self, picture_name, robot=None, image=None):
        self.picture_name = picture_name

        #if no image parameter is provided image will be taken from pictures_balls\<picture_name>
        if image is None:
            self.picture = cv2.imread('pictures_balls\\' + self.picture_name)
            self.gray_picture = cv2.imread('pictures_balls\\' + self.picture_name, 0)
        else:
            self.picture = image
            self.gray_picture = cv2.cvtColor(self.picture, cv2.COLOR_BGR2GRAY)

        self.hsv_image = cv2.cvtColor(self.picture, cv2.COLOR_BGR2HSV)
        self.robot = robot

    def show_picture(self):
        picture = cv2.cvtColor(self.picture, cv2.COLOR_BGR2RGB)
        cv2.imshow("picture", picture)

    def show_gray(self):
        cv2.imshow("gray picture", self.gray_picture)

    def create_mask(self, lower_array_DOWN, upper_array_DOWN, lower_array_UP, upper_array_UP, image):
        # lower mask (0-10)
        mask0 = cv2.inRange(image, lower_array_DOWN, upper_array_DOWN)

        # upper mask (170-180)
        mask1 = cv2.inRange(image, lower_array_UP, upper_array_UP)

        mask = mask0 + mask1
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        return mask

    def single_mask(self, lower, upper, image):
        mask = cv2.inRange(image, lower, upper)

        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)

        return mask

    def get_blob_info(self, keypoints):
        messages = []
        for keyPoint in keypoints:
            rows = self.picture.shape[0]
            cols = self.picture.shape[1]

            center_x = cols//2
            center_y = rows//2

            x_key = int(keyPoint.pt[0])
            y_key = int(keyPoint.pt[1])

            message = ''


            if  (x_key == center_x or (center_x - 10 <= x_key <= center_x + 10))\
            and (y_key == center_y or (center_y + 10 >= y_key >= center_y - 10)):
                message += 'One is in the middle '

            else:
                if   x_key <= center_x - 10: message += 'One is to the left '
                elif x_key >= center_x + 10: message += 'One is to the right '

                if   y_key <= center_y - 10: message += 'and close '
                elif x_key >= center_x + 10: message += 'and far '

            messages.append(message)

        return messages

    def detect_blobs(self, mask, picture_for_keypoints):
        detector = cv2.SimpleBlobDetector_create(params_setter())
        reverse_mask = 255 - mask

        keypoints = detector.detect(reverse_mask)

        # draw circle, around
        image_with_circles = cv2.drawKeypoints(picture_for_keypoints, keypoints, np.array([]), (123, 255, 100),
                                               cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

        # draw centers of objects
        centers = cv2.drawKeypoints(image_with_circles, keypoints, np.array([]), (123, 255, 100),
                                    cv2.DFT_REAL_OUTPUT)

        return centers, len(keypoints), self.get_blob_info(keypoints)

    def detect_all_colors(self):
        colors = {
            'red':    self.create_mask(np.array([10, 50, 50]), np.array([10, 255, 255]),
                                       np.array([170, 50, 55]), np.array([180, 255, 255]), self.hsv_image),

            'yellow': self.single_mask(np.array([30, 100, 100]), np.array([40, 255, 255]), self.hsv_image),

            'blue':   self.single_mask(np.array([108, 57, 38]), np.array([120, 255, 255]), self.hsv_image),

            'green':  self.single_mask(np.array([67, 55, 40]), np.array([(70, 255, 255)]), self.hsv_image)
        }

        picture_with_keypoints = self.picture

        for colour in colors:
            picture_with_keypoints, number, messages = self.detect_blobs(colors[colour], picture_with_keypoints)
            if number > 0:
                say = 'There '

                if number > 1: say += ' are ' + str(number) + ' ' + colour + ' balls:\n'
                else: say += 'is ' + str(number) + ' ' + colour + ' ball:\n'

                for message in messages:
                    say += message + '\n'

                robot.set_english()
                robot.say_text(say)
                print(say)

        cv2.imshow("final", picture_with_keypoints)
        cv2.waitKey(0)


    def camera_detect(self):
        camera = cv2.VideoCapture(0)

        while True:
            _, frame = camera.read()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = self.create_mask(np.array([0, 50, 50]), np.array([10, 255, 255]),
                                       np.array([170, 50, 50]), np.array([180, 255, 255]), frame)

            red_bitwise = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Frame", frame)
            cv2.imshow("Red mask", red_bitwise)

            key = cv2.waitKey(1)
            if key == 27:
                break
