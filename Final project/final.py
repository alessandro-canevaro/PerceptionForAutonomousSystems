import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

rect_map_left_x = np.load(r'matrix_calib_rectify\map_left_x.npy')
rect_map_left_y = np.load(r'matrix_calib_rectify\map_left_y.npy')
rect_map_right_x = np.load(r'matrix_calib_rectify\map_right_x.npy')
rect_map_right_y = np.load(r'matrix_calib_rectify\map_right_y.npy')

imgs_names_left = glob.glob('./Stereo_conveyor_with_occlusions/left/*.png')
imgs_names_right = glob.glob('./Stereo_conveyor_without_occlusions/right/*.png')

def beltmask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_not(cv2.inRange(hsv, np.array([100,50,50]), np.array([115,255,255])))

def findobjectbounds(frame, bgs):
    fgMask = bgs.apply(frame)

    kernel = np.ones((5,5), np.uint8) 
    fgMask = cv2.dilate(fgMask, kernel, iterations=1) 
    fgMask = cv2.erode(fgMask, kernel, iterations=1) 

    contours, _ = cv2.findContours(fgMask,2,1)         

    pts = np.array([[410, 490],[1110,310],[1250,370],[460,660]], np.int32)
    pts = pts.reshape((-1,1,2))

    for c in reversed(sorted(contours, key=cv2.contourArea)):
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
        if radius > 30 and radius < 200:
            if cv2.pointPolygonTest(pts, (center_x, center_y), False) > 0:
                return True, center_x, center_y, radius
    return False, 0, 0, 0

img_left = cv2.imread(imgs_names_left[0])
img_left = cv2.remap(img_left, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
backSub = cv2.createBackgroundSubtractorMOG2(3, 16, False)
_ = backSub.apply(img_left)

measure_history = []
prediction_history = []
status_history = ['approaching']
founded = False

start, end = 100, 750
for frame_counter, (img_name_left, img_name_right) in enumerate(zip(imgs_names_left[start:end], imgs_names_right[start:end])):

    img_left = cv2.imread(img_name_left)
    img_left = cv2.remap(img_left, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
    img_output = img_left.copy()

    
    belt_mask = beltmask(img_output)
    result = cv2.bitwise_and(img_output, img_output, mask=belt_mask)
    founded, center_x, center_y, radius = findobjectbounds(result, backSub)

    measure_history.append((center_x, center_y, radius))

    if founded:
        cv2.circle(img_output, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2) #cirle
        cv2.circle(img_output, (int(center_x), int(center_y)), 5, (0, 0, 255), -1) #center

        if center_x > 1200 and status_history[-1] == 'leaving':
            status_history.append('approaching') 
        elif center_x > 1090 and status_history[-1] == 'approaching':
            status_history.append('at start') 
        elif center_x < 1090 and status_history[-1] == 'at start':
            status_history.append('in middle') 
        elif center_x < 720 and status_history[-1] == 'in middle':
            status_history.append('at end') 
        elif center_x < 470 and status_history[-1] == 'at end':
            status_history.append('leaving') 
    else:
        status_history.append(status_history[-1])

    print(status_history[-1])

    cv2.line(img_output,(1200,0),(1200,720),(255,0,0),1)
    cv2.line(img_output,(1090,0),(1090,720),(255,0,0),1)
    cv2.line(img_output,(720,0),(720,720),(255,0,0),1)
    cv2.line(img_output,(470,0),(470,720),(255,0,0),1)
    cv2.imshow('Output', img_output)
    cv2.waitKey(50)

cv2.destroyAllWindows()

x_meas = [i[0] for i in measure_history]
y_meas = [i[1] for i in measure_history]
r_meas = [i[2] for i in measure_history]
#x_pred = [i[0] for i in prediction_history]
#y_pred = [i[1] for i in prediction_history]
#z_pred = [i[2] for i in prediction_history]
#plt.scatter(x_meas, y_meas)
#plt.scatter(x_pred, y_pred)
plt.plot(x_meas)
#plt.plot(x_pred)
plt.figure()
plt.plot(y_meas)
#plt.plot(y_pred)
plt.figure()
plt.plot(r_meas)
#plt.plot(z_pred)
#plt.figure()
#plt.plot(radius_history)
plt.show()