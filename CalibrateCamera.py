import numpy
import sys
import cv2
from cv2 import aruco
import pickle
import glob
from time import time, sleep
import numpy as np
import json
from datetime import datetime

if len(sys.argv)>1:
        camera_id = int(sys.argv[1])
else:
        camera_id = 1

date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
CHARUCOBOARD_ROWCOUNT = 4
CHARUCOBOARD_COLCOUNT = 6

ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
print(date)
# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.190,
        markerLength=0.148,
        dictionary=ARUCO_DICT)



# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
image_size = None # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = 0
#camera_intrinsec = cv2.VideoCapture("./output/intrinsec_cam{}.avi".format(camera_id))
#camera_extrinsec = cv2.VideoCapture("./output/extrinsec_cam{}.avi".format(camera_id))
# Loop through images glob'ed
#crop = np.array((153,478,228,558))
print('camera :',camera_id)
while images < 240:
    img = cv2.imread('./calibration_img/cam{}/intrinsic/img{}.png'.format(camera_id,images+1))
    #grab, img = camera_intrinsec.read()
    #if not grab:break
    #if images == 243: continue
    #proportion = max(img.shape) / 1000.0
    # Open the image
    images += 1
    #if images%2!=0: continue
    print('intrinsic', images)
    # print(images)
    # Grayscale the image
    # x1,y1,x2,y2 = (crop*proportion).astype(int)
    
    # img[y1:y2, x1:x2] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Charuco board', gray)
    #cv2.waitKey(0)
    # Find aruco markers in the query image
    corners, ids, _ = cv2.aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    img = cv2.aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    # Get charuco corners and ids from detected aruco markers
    if ids is None: continue
    response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares
    print(response)
    if response > 14:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)
        
        # # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = aruco.drawDetectedCornersCharuco(
                 image=img,
                 charucoCorners=charuco_corners,
                 charucoIds=charuco_ids)
      
        # # If our image size is unknown, set it now
        if not image_size:
            image_size = gray.shape[::-1]
    
        # # Reproportion the image, maxing width or height at 1000
        # img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
        # # Pause to display each image, waiting for key press
        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
  
    # sleep(1)

  # Destroy any open CV windows
cv2.destroyAllWindows()
print("num_images: {}".format(len(corners_all)))
# Make sure at least one image was found
if images < 1:
    # Calibration failed because there were no images, warn the user
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)
    
#print(cameraMatrix.reshape(-1).tolist())
  # Print matrix and distortion coefficient to the console
print(f"cameraMatrix = {cameraMatrix}")
print(f"distCoeffs = {distCoeffs}")
print(f"calibration = {calibration_error}")
      
  # # Save values to be used where matrix+dist is required, for instance for posture estimation
  # # I save files in a pickle file, but you can use yaml or whatever works for you
  # f = open('calibration.pckl', 'wb')
  # pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
  # f.close()
markerSize = 0.4
#arucoParams = cv2.aruco.DetectorParameters_create()
images = 0
image_h = 720
image_w = 1280
while images < 17:
  #grab, img = camera_extrinsec.read()
  img = cv2.imread('./calibration_img/cam1/extrinsic/img{}.png'.format(images+1))
  #if not grab:break
  # Open the image
  images += 1
  if images%1!=0:continue
  print('extrinsecos:',images)

  (image_h,image_w) = img.shape[:2]

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
  rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSize, cameraMatrix, distCoeffs)
  cv2.aruco.drawDetectedMarkers(img, corners, ids)
  # cv2.imshow('Aruco', img)
  # cv2.waitKey(4)
  marker_id = np.where(ids.reshape(-1)==7)
  rvec = rvec[marker_id].reshape((1,-1))
  print(rvec.shape, ids, rvec.shape)
  rotation,_ = cv2.Rodrigues(rvec)
  translation = tvec[marker_id].reshape((3,1))
  print("//",rotation.shape, translation.shape, np.array([[0, 0, 0, 1]]).shape)
  extrinsecs = cv2.hconcat([rotation, translation])
  print(extrinsecs.shape)
  last_colum = np.zeros((1,4))
  last_colum[:,-1] = 1
  extrinsecs = cv2.vconcat([extrinsecs, last_colum])
  print(extrinsecs.shape, cameraMatrix.shape, distCoeffs.shape)
  break

  # Print to console our success
  # print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))

calibration_parameters = {
  "error": calibration_error,
  "resolution": {
    "height": image_h,
    "width": image_w,
  },
  "id": "{}".format(camera_id),
  "extrinsic": {
    "to": "{}".format(camera_id),
    "tf": {
      "shape": {
        "dims": [
          {
            "name": "rows",
            "size": 4
          },
          {
            "name": "cols",
            "size": 4
          }
        ]
      },
      "type": "DOUBLE_TYPE",
      "doubles": extrinsecs.reshape(-1).tolist()
    },
    "from": "1000"
  },
  "calibratedAt": "2021-09-27T16:41:19.396796441Z",
  "intrinsic": {
    "shape": {
      "dims": [
        {
          "name": "rows",
          "size": cameraMatrix.shape[0]
        },
        {
          "name": "cols",
          "size": cameraMatrix.shape[1]
        }
      ]
    },
    "type": "DOUBLE_TYPE",
    "doubles": cameraMatrix.reshape(-1).tolist()
  },
  "distortion": {
    "shape": {
      "dims": [
        {
          "name": "rows",
          "size": distCoeffs.shape[0]
        },
        {
          "name": "cols",
          "size": distCoeffs.shape[1]
        }
      ]
    },
    "type": "DOUBLE_TYPE",
    "doubles": distCoeffs.reshape(-1).tolist()
  }
}

with open('params_camera{}.json'.format(camera_id), 'w') as f:
  json.dump(calibration_parameters, f)


print(calibration_parameters)
