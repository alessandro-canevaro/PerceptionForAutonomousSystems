import cv2
import numpy as np
import glob

def beltmask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_not(cv2.inRange(hsv, np.array([100,50,50]), np.array([115,255,255])))

def objectmask(frame, bgs):
    fgMask = bgs.apply(frame)

    kernel = np.ones((5,5), np.uint8) 

    fgMask = cv2.dilate(fgMask, kernel, iterations=1) 

    contours, _ = cv2.findContours(fgMask,2,1)            
    contours = sorted(contours, key=cv2.contourArea)            
    out_mask = np.zeros_like(fgMask)
    cv2.drawContours(out_mask, [contours[-1]], -1, 255, cv2.FILLED, 1)  
    return out_mask

def findobjectbounds(mask):
    contours, _ = cv2.findContours(mask,2,1) 
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
        return center_x, center_y, radius
    else:
        return 0, 0, 0
    
def computeDisparity(left, right): #left and right undistoreted and rectified images (3 channels)
    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=int((1280 / 8) + 15) & -16,#160
                                   blockSize=5,
                                   P1=8*3*5**2, #600
                                   P2=32*3*5**2, #2400
                                   disp12MaxDiff=100,
                                   preFilterCap=32,
                                   uniquenessRatio=10,
                                   speckleWindowSize=0,
                                   speckleRange=32)
    disp = stereo.compute(left, right)
    disp = cv2.medianBlur(disp, 5)
    return disp

def getDepth(x_center, y_center, disparity, focal_lenght=685, baseline=120):
    depth_list = [0]#if depth is not found zero is returned
    for x in [x_center-5, x_center, x_center+5]:
        for y in [y_center-5, y_center, y_center+5]:
            depth_list.append(focal_lenght*baseline/(disparity[y,x]/16.0)) #in mm
    return max(depth_list)

rect_map_left_x = np.load(r'matrix_calib_rectify\map_left_x.npy')
rect_map_left_y = np.load(r'matrix_calib_rectify\map_left_y.npy')
rect_map_right_x = np.load(r'matrix_calib_rectify\map_right_x.npy')
rect_map_right_y = np.load(r'matrix_calib_rectify\map_right_y.npy')
#imgL = cv2.remap(imgL, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
#imgR = cv2.remap(imgR, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)

leftImages = glob.glob('./Stereo_conveyor_with_occlusions/left/*.png')
rightImages = glob.glob('./Stereo_conveyor_with_occlusions/right/*.png')
frame = cv2.imread(leftImages[0])
frame = cv2.remap(frame, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)

backSub = cv2.createBackgroundSubtractorMOG2(30, 16, False)
_ = backSub.apply(frame)

prev_center_x, prev_center_y = 0, 0
measure_history = []
prediction_history = []
radius_history = []
center_on_belt = False
prev_center_on_belt = False
initialized = False
start, end = 100, -1
counter = 0
for i, (imgL, imgR) in enumerate(zip(leftImages[start:end], rightImages[start:end])):
    counter += 1
    frame = cv2.imread(imgL)
    frame = cv2.remap(frame, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
    rect_left = frame.copy()

    update_filter=False
    center_x, center_y, center_z = 0, 0, 0
    
    object_mask = objectmask(frame, backSub)
    belt_mask = beltmask(frame)

    result = cv2.bitwise_and(object_mask, object_mask, mask=belt_mask)
    
    center_x, center_y, radius = findobjectbounds(result)
    
    
    pts = np.array([[410, 490],[1110,310],[1250,370],[460,660]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(frame,[pts],True,(255,0,255))
    #cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 255), -1) #center
    if cv2.pointPolygonTest(pts, (center_x, center_y), False) < 0:
        #print(counter, "not on belt", center_x, center_y)
        center_x, center_y, center_z = 0, 0, 0
        center_on_belt = False
    else:
        #print(counter, "on belt", center_x, center_y)
        center_on_belt = True
        
    prev_center_x, prev_center_y = center_x, center_y
    if cv2.norm(np.array([[center_x, center_y]]), np.array([[prev_center_x, prev_center_y]])) > 20:
        center_x, center_y, center_z = 0, 0, 0
        
    if radius < 30 or radius > 200:
        center_x, center_y, center_z = 0, 0, 0
        
    for c_x, c_y, c_z in measure_history: #draws previous centers
        cv2.circle(frame, (int(c_x), int(c_y)), 1, (0, 255, 255), -1) #center
        
    if center_on_belt and not prev_center_on_belt:
        kalman = cv2.KalmanFilter(6, 3, 0)
        kalman.transitionMatrix = np.array([[1, 1, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0],
                                            [0, 0, 1, 1, 0, 0],
                                            [0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 1, 1],
                                            [0, 0, 0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = 1e-5 * np.eye(6, dtype=np.float32)
        kalman.measurementNoiseCov = 1e-2 * np.eye(3, dtype=np.float32)
        kalman.errorCovPost = 1. * np.eye(6, dtype=np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                             [0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0]], np.float32)
        kalman.statePre = np.array([[center_x], [-1], [center_y], [1], [center_z], [0]], np.float32)
        initialized = True
        print("initialized", counter)

    if center_x != 0 and center_y != 0:
        frame_right = cv2.imread(imgR)
        rect_right = cv2.remap(frame_right, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)
        #disparity = computeDisparity(rect_left, rect_right)
        center_z = 10#getDepth(int(center_x), int(center_y), disparity)
        #print(center_z)
        #disparity = cv2.normalize(disparity, disparity, alpha=255,
        #                      beta=0, norm_type=cv2.NORM_MINMAX)
        #disparity = np.uint8(disparity)
        #cv2.imshow('disp', disparity)
        if center_z > 0:

            frame = cv2.addWeighted(frame, 1, cv2.bitwise_and(frame,frame,mask = result), 1, 0)
            cv2.circle(frame, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2) #cirle
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1) #center

            z = np.array([[center_x],
                          [center_y],
                          [center_z]], dtype=np.float32)
            #print(z)
            #print("updated")
            kalman.correct(z)
    else:
        pass
        #print("not detected")

    if initialized:
        ### Predict the next state
        x = kalman.predict()
        #print(x)

        ### Draw the current tracked state and the predicted state on the image frame ###
        x_pred, y_pred, z_pred = np.matmul(kalman.measurementMatrix, x).ravel()
        prediction_history.append((x_pred, y_pred, z_pred))
        cv2.circle(frame,(int(x_pred), int(y_pred)), 5, (255,0,0),-1)
    else:
        prediction_history.append((0, 0, 0))

    radius_history.append(radius)
    measure_history.append((center_x, center_y, center_z))
    
    prev_center_on_belt = center_on_belt
    
    # Show the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(50)

cv2.destroyAllWindows()


import matplotlib.pyplot as plt
x_meas = [i[0] for i in measure_history]
y_meas = [i[1] for i in measure_history]
z_meas = [i[2] for i in measure_history]
x_pred = [i[0] for i in prediction_history]
y_pred = [i[1] for i in prediction_history]
z_pred = [i[2] for i in prediction_history]
#plt.scatter(x_meas, y_meas)
#plt.scatter(x_pred, y_pred)
plt.plot(x_meas)
plt.plot(x_pred)
plt.figure()
plt.plot(y_meas)
plt.plot(y_pred)
plt.figure()
plt.plot(z_meas)
plt.plot(z_pred)
plt.figure()
plt.plot(radius_history)
plt.show()