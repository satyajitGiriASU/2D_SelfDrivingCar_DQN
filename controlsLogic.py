def checkifinlane(width,car_loc,list_other_loc):
	#if there is car within yourCarX +- carWidth/2
	inLaneFlag = False
	for locn in list_other_loc:
		if abs(locn[0]-car_loc[0])<(width):
			# print "locn = ", locn, "diff = " abs(locn[0]-car_loc[0]), "width = ", width
			inLaneFlag = True
			break
	return inLaneFlag

def carInLane(width,car_loc,list_other_loc):
	minDist = None
	minIndx = None
	for indx, locn in enumerate(list_other_loc):
		if abs(locn[0]-car_loc[0])<(width):
			if minDist is None:
				minDist =  abs(locn[1]-car_loc[1])
				minIndx = indx
			else:
				if minDist > abs(locn[1]-car_loc[1]):
					minDist =  abs(locn[1]-car_loc[1])
					minIndx = indx
	return minIndx

def carRight(width, car_loc,list_other_loc):
	inRightFlag = False
	for locn in list_other_loc:
		if (locn[0]-car_loc[0])<(3*width) and (locn[0]-car_loc[0])>0:
			inRightFlag = True
			break
	return inRightFlag

def carLeft(width,car_loc,list_other_loc):
	inLeftFlag = False
	for locn in list_other_loc:
		if (locn[0]-car_loc[0])>(-3*width) and (locn[0]-car_loc[0])<0:
			inLeftFlag = True
			break
	return inLeftFlag
			



def controlsLogic(heigth, width,fps, car_loc, car_speed, list_other_loc, list_other_speed):
	carInLaneFLAG = checkifinlane(width,car_loc,list_other_loc)
	print  carInLaneFLAG, fps, car_loc, car_speed, list_other_loc, list_other_speed
	if not carInLaneFLAG and car_speed < 60:
		print "No Car in lane and speed less than 60"
		dd_backImgScrollSpeed = 1
		carPosXChange = 0
	elif not carInLaneFLAG and car_speed >= 60 and car_speed <= 65:
		print "No Car in lane and speed greater than 60"
		dd_backImgScrollSpeed = 0
		carPosXChange = 0
	elif not carInLaneFLAG and car_speed >= 65:
		print "No Car in lane and speed greater than 60"
		dd_backImgScrollSpeed = -.25
		carPosXChange = 0
	elif carInLaneFLAG:
		carInFrontIndex = carInLane(width,car_loc,list_other_loc)
		carInFrontX = list_other_loc[carInFrontIndex][0]
		carInFrontY = list_other_loc[carInFrontIndex][1]
		carInFrontSpeed = list_other_speed[carInFrontIndex]

		if car_speed>=30:
			print "Car is in lane and our is speed greater than 30"
			#distance seperating the two cars the car height and some buffer
			dist = (car_loc[1] - carInFrontY - heigth)*.8

			#d = ut + .5*a*t^2
			# dd_backImgScrollSpeed = -2 * (dist - (car_speed-30)*(fps)) / (fps*fps)
			# dd_backImgScrollSpeed = -3
			dd_backImgScrollSpeed = (30-car_speed)/(fps*.9)
			print "dist = ", dist, "dd  = ", dd_backImgScrollSpeed
			carPosXChange = 0
		else:
			print "Car is in lane and our is speed less than 30"
			dd_backImgScrollSpeed = 0

			#is the car in left most lane?
			if car_loc[0] < (220 + 2 * width):
				carPosXChange = 8

			elif car_loc[0] > (980 - 2 * width):
				carPosXChange = -8

			#is there car in immediate left? if yes go right
			elif carLeft(width,car_loc,list_other_loc):
				carPosXChange = 8

			#is there car in immediate right? if yes go left
			if carRight(width,car_loc,list_other_loc):
				carPosXChange = -8

			#is there car in immediate left? if yes go right
			elif carLeft(width,car_loc,list_other_loc):
				carPosXChange = 8

			#if no to both go to the direction which has least amount of lane change necessary
			else:
				direction = (carInFrontX - car_loc[0])/abs(carInFrontX - car_loc[0])
				carPosXChange = direction * 8

	return dd_backImgScrollSpeed,carPosXChange








		



