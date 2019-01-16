# PyCar | Developed by Sirajuddin Asjad and Kirisan Manivannan

# INCLUDES
import pygame
from time import sleep, time
from random import randrange
from pygame.locals import *
from variables import *
import numpy as np
import math
import matplotlib.pyplot as plt
# import cv2
from controlsLogic import controlsLogic

# MAIN
pygame.init()
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption(title)
FPS = pygame.time.Clock()


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# FUNCTIONS
def printTimer(seconds):
	font = pygame.font.Font('freesansbold.ttf', 20)
	text = font.render("Time:  %i seconds" % int(seconds), True, black)
	screen.blit(text, (20, 20))

def countScore(count):
	font = pygame.font.Font('freesansbold.ttf', 20)
	text = font.render("Score: %i points" % int(count), True, black)
	screen.blit(text, (20, 45))

def printSpeed(count):
	font = pygame.font.Font('freesansbold.ttf', 20)
	text = font.render("Speed: %i points" % int(count), True, black)
	screen.blit(text, (20, 70))

def traffic(x, y):
	blueCar = pygame.image.load('inc/car2.png')	
	blueCarHeight = blueCar.get_rect().height
	blueCarWidth = blueCar.get_rect().width
	blueCar = pygame.transform.scale(blueCar, (int(blueCarWidth/2), int(blueCarHeight/2)))
	blueCar = blueCar.convert_alpha()

	screen.blit(blueCar, (x, y))

def car(x, y):
	carImg = pygame.image.load('inc/car.png')
	carImg = carImg.convert_alpha()
	carImgHeight = carImg.get_rect().height
	carImgWidth = carImg.get_rect().width
	carImg = pygame.transform.scale(carImg, (int(carImgWidth/2), int(carImgHeight/2)))
	screen.blit(carImg, (x, y))

def textprint(text, font):
	textSurface = font.render(text, True, black)
	return textSurface, textSurface.get_rect()

def displaytext(text):
	largeText = pygame.font.Font('freesansbold.ttf', 115)
	textSurface, textRectangle = textprint(text, largeText)
	textRectangle.center = ((screenWidth/2), (screenHeight/2))
	screen.blit(textSurface, textRectangle)
	pygame.display.flip()
	sleep(2)
	# main()

def crashCar():
	displaytext('GAME  OVER')

def background(y):
	backImg = pygame.image.load('inc/road.jpg')
	backImg = backImg.convert_alpha()
	backImgHeight = backImg.get_rect().height
	scrollY = y % backImgHeight
	screen.blit(backImg, (0, scrollY - backImgHeight))
	if scrollY < screenHeight:
		screen.blit(backImg, (0, scrollY))


def loader():
	loadTime = time()
	loading = True
	while loading:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		
		loadImg = pygame.image.load('inc/loading.png')
		loadImg = loadImg.convert_alpha()
		loadRectangle = loadImg.get_rect()
		screen.blit(loadImg, loadRectangle)
		pygame.display.flip()
		
		if time() > (loadTime + 7):
			screen.fill(white)
			pygame.display.flip()
			loading = False

# MAIN FUNCTION
def main():
	carPosX = 610
	carPosY = 575
	carPosXChange = 0
	backImgScrollSpeed = 0
	d_backImgScrollSpeed = 40
	dd_backImgScrollSpeed = 0
	trafficPosX = randrange(220, 980)
	trafficPosX1 = randrange(180, 900)
	trafficPosY = -600
	trafficPosY1 = - 300
	trafficSpeed = 30
	startTime = time()
	prevTime = startTime
	score = 0
	second_count = 0
	exit = False
	speed_vector = []
	frame_count = 0

	cImg = pygame.image.load('inc/car.png')
	cImg = cImg.convert_alpha()
	cImgHeight = cImg.get_rect().height
	cImgWidth = cImg.get_rect().width

	while not exit:

		# for event in pygame.event.get():
		# 	if event.type == pygame.QUIT:
		# 		return speed_vector
		# 		pygame.quit()
		# 		quit()

				
		# 	if event.type == pygame.KEYDOWN:
		# 		if event.key == pygame.K_LEFT:
		# 			carPosXChange = -12
		# 		elif event.key == pygame.K_RIGHT:
		# 			carPosXChange = 12
		# 		elif event.key == pygame.K_UP:
		# 			dd_backImgScrollSpeed = 1
		# 		elif event.key == pygame.K_DOWN:
		# 			dd_backImgScrollSpeed = -1
				
		# 	if event.type == pygame.KEYUP:
		# 		if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
		# 			carPosXChange = 0
		# 		if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
		# 			dd_backImgScrollSpeed = 0
		
		checkTime = time()	
		background(backImgScrollSpeed)

		if checkTime >= startTime:
			if d_backImgScrollSpeed < 0:
				dd_backImgScrollSpeed = 0
				d_backImgScrollSpeed = 0
			else:
				d_backImgScrollSpeed = d_backImgScrollSpeed + dd_backImgScrollSpeed


			backImgScrollSpeed = backImgScrollSpeed + d_backImgScrollSpeed
			# trafficSpeed = max(0,d_backImgScrollSpeed-30)
			trafficSpeed = d_backImgScrollSpeed-30

		
		frame_count += 1	
		carPosX = carPosX + carPosXChange
		traffic(trafficPosX, trafficPosY)
		traffic(trafficPosX1, trafficPosY1)
		trafficPosY = trafficPosY + trafficSpeed
		trafficPosY1 = trafficPosY1 + trafficSpeed
		car(carPosX, carPosY)
		printTimer(round(checkTime - startTime))
		countScore(score)
		printSpeed(d_backImgScrollSpeed)

		if carPosX > 980 or carPosX < 220:
			crashCar()
			return speed_vector

		if trafficPosY > screenHeight:
			trafficPosY = 0 - carHeight
			trafficPosX = randrange(220, 980)
			score = score + 1

		if trafficPosY1 > screenHeight:
			trafficPosY1 = 0 - carHeight
			trafficPosX1 = randrange(220, 980)
			score = score + 1

		if carPosY < (trafficPosY + carHeight):
			if carPosX > trafficPosX and carPosX < (trafficPosX + carWidth) or (carPosX + carWidth) > trafficPosX and (carPosX + carWidth) < (trafficPosX + carWidth):
				print "Crash Happened with car 1"
				crashCar()
				print "Now we should exit"
				print "Speed of Crashed Car = ", d_backImgScrollSpeed
				print "Frame Rate = ", frame_count*1.0/(second_count+1)
				return speed_vector

		if carPosY < (trafficPosY1 + carHeight):
			if carPosX > trafficPosX1 and carPosX < (trafficPosX1 + carWidth) or (carPosX + carWidth) > trafficPosX1 and (carPosX + carWidth) < (trafficPosX1 + carWidth):
				print "Crash Happened with car 2"
				crashCar()
				print "Now we should exit"
				print "Speed of Crashed Car = ", d_backImgScrollSpeed
				print "Frame Rate = ", frame_count*1.0/(second_count+1)
				return speed_vector

		if checkTime-prevTime > .25:
			

			prevTime = checkTime
			speed_vector.append(d_backImgScrollSpeed)
			# screen_frame = pygame.display.get_surface()
			# # # pygame.image.save(screen,'lol3.png')
			# imgdata = pygame.surfarray.array3d(screen_frame)
			# imgdata = imgdata[...,::-1]
			# imgdata = cv2.flip(imgdata,0)
			# imgdata = np.rot90(imgdata,3)
			# # cv2.imwrite('lolfinal.png',imgdata)
			# # exit()
			# out.write(imgdata)


			second_count += 1
			avg_frame_rate = frame_count*1.0/second_count
			dd_backImgScrollSpeed,carPosXChange = controlsLogic(cImgHeight*1.0/2, cImgWidth*1.0/2, avg_frame_rate,(carPosX,carPosY),d_backImgScrollSpeed,[(trafficPosX,trafficPosY),(trafficPosX1,trafficPosY1)],[min(d_backImgScrollSpeed,30),min(d_backImgScrollSpeed,30)])
			print "xChange = ", carPosXChange, "dd = ", dd_backImgScrollSpeed

			if second_count >= 120:
				return speed_vector

		pygame.display.flip()
		FPS.tick(500)

no_iterations = 1
speed_values = np.zeros((no_iterations,120))
dist_accumalated = np.zeros((no_iterations,120))
speed_values2 = np.ones(speed_values.shape)*40


for i in range(no_iterations):
	loader()
	speed_list = main()
	speed_vector = np.asarray(speed_list)
	speed_values[i,:len(speed_vector)] = speed_vector
	pygame.quit()
	

dist_values = speed_values/3.6

for row in range(dist_values.shape[0]):
	for col in range(dist_values.shape[1]):
		dist_accumalated[row,col] = sum(dist_values[row,:col+1])

print("Speed = ", speed_values)
print("Distance Covered in that time interval = ", dist_values)
print("Total Distance covered = ", dist_accumalated)


speed_values2[:,1:] = speed_values[:,:-1]
acceleration_values = (speed_values-speed_values2) * 1000/3600
print("Acceleration = ", acceleration_values)


mass_vehicle = 2000 #kg
mass_rolling_inertia = 0 #kg
gravity = 9.8 #ms-2
rolling_resistance = 0.011 #-
road_gradient_angle = 0 #degrees
air_density = 1.2 #kg/m^3
drag_coeff = 0.277
vehicular_equivalent_crossection = .9290304 #m^2


term1 = (mass_vehicle*gravity*(rolling_resistance*math.cos(road_gradient_angle)+math.sin(road_gradient_angle))) 
term2 = (0.0386 * air_density * drag_coeff * vehicular_equivalent_crossection) + np.square(speed_values)

term3 = (mass_vehicle + mass_rolling_inertia) * abs(acceleration_values)

energy_values = np.multiply((1.0/3600)*(term1+term2+term3),dist_values)

print("energy_values", energy_values)	

print ("speed_values[0]",speed_values[0])
plt.plot(speed_values[0])
plt.xlabel('Time(s)')
plt.ylabel("Speed values(km/hr)")
plt.title("Plot of speed vs time")
plt.show()
plt.plot(dist_accumalated[0])
plt.xlabel("Time(s)")
plt.ylabel("Distance values(m)")
plt.title("Plot of distance vs time")
plt.show()
plt.plot(energy_values[0])
plt.xlabel("Time(s)")
plt.ylabel("Energy values(kWh)")
plt.title("Plot of energy vs time")
# plt(dist_values)
# plt(acceleration_values)
# plt(energy_values)
plt.show()

# out.release()



quit()