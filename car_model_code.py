
import pygame
from time import sleep, time
from random import randrange
from pygame.locals import *
from variables import *
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import time 

pygame.init()
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption(title)
FPS = pygame.time.Clock()


class car_model:
    # initalize the parameters
    def __init__(self):
        self.startTime  = time.time()
        self.carPosX = 610
        self.carPosY = 575
        self.carPosXChange = -12
        self.backImgScrollSpeed = 0
        self.d_backImgScrollSpeed = 40
        self.dd_backImgScrollSpeed = 0
        self.trafficPosX = randrange(220, 980)
        self.trafficPosX1 = randrange(180, 900)
        self.trafficPosY = -600
        self.trafficPosY1 = - 300
        self.trafficSpeed = 30
        self.score = 0
        self.second_count = 0
        self.speed_vector = []
        self.frame_count = 0
        self.screenWidth = 1280
        self.screenHeight = 720
        # CAR IMAGE RESOLUTION
        self.carWidth = 53
        self.carHeight = 97
        cImg = pygame.image.load('inc/car.png')
        self.cImg =  cImg.convert_alpha()
        self.cImgHeight = cImg.get_rect().height
        self.cImgWidth = cImg.get_rect().width


        self.background(self.backImgScrollSpeed)

        self.countScore(self.score)
        car_1 = self.car(self.carPosX, self.carPosY)
        traffic_1 = self.traffic(self.trafficPosX,self.trafficPosY)
        
        # next_state1 = next_state(self,4)
        # pygame.display.update()
    
    def printTimer(self,seconds):
        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render("Time:  %i seconds" % int(seconds), True, black)
        screen.blit(text, (20, 20)) 

    def countScore(self,count):
        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render("Score: %i points" % int(count), True, black)
        screen.blit(text, (20, 45)) 

    def printSpeed(self, count):
        font = pygame.font.Font('freesansbold.ttf', 20)
        text = font.render("Speed: %i units" % int(count), True, black)
        screen.blit(text, (20, 70))

    def background(self,y):
        # print ("displaying background")
        backImg = pygame.image.load('inc/road.jpg')
        backImg = backImg.convert_alpha()
        backImgHeight = backImg.get_rect().height
        scrollY = y % backImgHeight
        screen.blit(backImg, (0, scrollY - backImgHeight))
        if scrollY < screenHeight:
            screen.blit(backImg, (0, scrollY))

    def traffic(self, x, y):
        # print (" generating traffic")
        blueCar = pygame.image.load('inc/car2.png') 
        blueCarHeight = blueCar.get_rect().height
        blueCarWidth = blueCar.get_rect().width
        blueCar = pygame.transform.scale(blueCar, (int(blueCarWidth/2), int(blueCarHeight/2)))
        blueCar = blueCar.convert_alpha()
        screen.blit(blueCar, (x, y))
        

    def car(self, x, y):
        # print ("displaying car")
        carImg = pygame.image.load('inc/car.png')
        carImg = carImg.convert_alpha()
        carImgHeight = carImg.get_rect().height
        carImgWidth = carImg.get_rect().width
        carImg = pygame.transform.scale(carImg, (int(carImgWidth/2), int(carImgHeight/2)))
        screen.blit(carImg, (x, y))

    # function for car crash
    def did_car_crash(self):
        if self.carPosY < (self.trafficPosY + self.carHeight):
            if self.carPosX > self.trafficPosX and self.carPosX < (self.trafficPosX + self.carWidth) or (self.carPosX + self.carWidth) > self.trafficPosX and (self.carPosX + self.carWidth) < (self.trafficPosX + self.carWidth):
                print "Crash Happened with car 1"
                print "Now we should exit"
                print "Speed of Crashed Car = ", self.d_backImgScrollSpeed
                return True
            

        if self.carPosY < (self.trafficPosY1 + self.carHeight):
            if self.carPosX > self.trafficPosX1 and self.carPosX < (self.trafficPosX1 + self.carWidth) or (self.carPosX + self.carWidth) > self.trafficPosX1 and (self.carPosX + self.carWidth) < (self.trafficPosX1 + self.carWidth):
                print "Crash Happened with car 2"
                print "Now we should exit"
                print "Speed of Crashed Car = ", self.d_backImgScrollSpeed
                return True
            
        if self.carPosX > 980 or self.carPosX < 220:
            return True
            
        return False

    # function to calculate reward
    def get_reward(self,terminate):
        reward = 0
        if(terminate):
            reward += -1000
        else:
            if ((self.d_backImgScrollSpeed < 90 and self.d_backImgScrollSpeed >30 )or self.d_backImgScrollSpeed < 0 ):
                reward = reward + (self.d_backImgScrollSpeed*1.0 /10) + min(self.dd_backImgScrollSpeed*10, 0)
            else:
                reward = reward - (self.d_backImgScrollSpeed*1.0 /10) + min(self.dd_backImgScrollSpeed*10, 0)
        if self.scoreIncreaseFlag:
            reward += 50
        return reward
    # converts pygame frame to cv2 numpy array
    def grab_frame(self):
        screen_frame = pygame.display.get_surface()
        imgdata = pygame.surfarray.array3d(screen_frame)
        imgdata = imgdata[...,::-1]
        imgdata = cv2.flip(imgdata,0)
        imgdata = np.rot90(imgdata,3)
        return imgdata

    # input is action, simulates the next state due to action
    def frame_step (self,action):
        self.scoreIncreaseFlag = False
        
        
        
        
        if (action[0] == 1):
            # do nothing
            self.dd_backImgScrollSpeed = 0
            self.carPosXChange = 0
            
        
        if(action[1] == 1):
            # medium brake
            self.dd_backImgScrollSpeed = -2
            self.carPosXChange = 0
            
        
        if(action[2] == 1):
            # brake hard
            self.dd_backImgScrollSpeed = -5
            self.carPosXChange = 0
            
        
        if(action[3] == 1):
            # speed up
            self.dd_backImgScrollSpeed = 3
            self.carPosXChange = 0
            
            

        
        if (action[4] == 1): 
            # go right
            self.dd_backImgScrollSpeed = 0
            self.carPosXChange = 12
            
        
        if (action[5] == 1):
            #  go left
            self.dd_backImgScrollSpeed = 0
            self.carPosXChange = -12

        if(action[6] == 1):
            # brake slow
            self.dd_backImgScrollSpeed = -1
            self.carPosXChange = 0
            
        self.d_backImgScrollSpeed = self.d_backImgScrollSpeed + self.dd_backImgScrollSpeed

        if self.d_backImgScrollSpeed < 0:
            self.dd_backImgScrollSpeed = 0
            self.d_backImgScrollSpeed = 0

        checkTime = time.time()
        
        # moving the traffic
        self.background(self.backImgScrollSpeed)
        self.backImgScrollSpeed = self.backImgScrollSpeed + self.d_backImgScrollSpeed
        self.printSpeed(self.d_backImgScrollSpeed)
        
        self.printTimer(round(checkTime - self.startTime))
        self.trafficSpeed = self.d_backImgScrollSpeed-30
        # moving the car
        self.carPosX = self.carPosX + self.carPosXChange
        self.trafficPosY = self.trafficPosY + self.trafficSpeed
        # initializing the traffic
        if self.trafficPosY > self.screenHeight:
            self.trafficPosY = 0 - self.carHeight
            self.trafficPosX = randrange(220, 980)
            self.score = self.score+1
            self.scoreIncreaseFlag = True
            
        self.trafficPosY1 = self.trafficPosY1 + self.trafficSpeed
        if self.trafficPosY1 > self.screenHeight:
            self.trafficPosY1 = 0 - carHeight
            self.trafficPosX1 = randrange(220, 980)
            self.score = self.score+1
            self.scoreIncreaseFlag = True

        #reinitialize traffic to beginning if car is going too slow
        if self.trafficPosY < -700:
            self.trafficPosY = -600 
        if self.trafficPosY1 < -400:
            self.trafficPosY1 = -300
        # find score
        self.countScore(self.score)
        # render traffic and car
        self.traffic(self.trafficPosX, self.trafficPosY)

        self.traffic(self.trafficPosX1, self.trafficPosY1)
        self.car(self.carPosX, self.carPosY)
        #update display
        pygame.display.update()
        # check if car crashed
        terminate = self.did_car_crash()
         # self.__init__()
        # calculate reward
        reward_val = self.get_reward(terminate)
        frame = self.grab_frame()
        # if crash reinitialize
        if(terminate):
            self.__init__()
        # image = self.grab_frame()
        return frame, reward_val,terminate



