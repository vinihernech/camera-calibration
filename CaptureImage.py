from is_wire.core import Channel,Subscription,Message
from is_msgs.image_pb2 import Image
import numpy as np
import cv2
import json
import time
import glob

def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

if __name__ == '__main__':
    #alterar o ID da camera e o diretório 
    print('---RUNNING EXAMPLE DEMO OF THE CAMERA CLIENT---')
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    broker_uri = "amqp://10.10.3.188:30000"
    camera_id = 6
    channel = Channel(broker_uri)
    subscription = Subscription(channel=channel,name="Intelbras_Camera")
    subscription.subscribe(topic='CameraGateway.{}.Frame'.format(1))

    window = 'Intelbras Camera'
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", 720, 1280)
    n=0
    img_array = []
    while True:
        msg = channel.consume()  
        im = msg.unpack(Image)
        frame = to_np(im)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            n = n+1
            img_array.append(frame)
            cv2.imwrite(f'./calibration_img/cam{camera_id}/intrinsic/img{n}.png',frame)
        elif cv2.waitKey(1) & 0xFF == ord('q'):       
            break
        cv2.imshow("camera", frame)
    
    # for filename in glob.glob(f'./calibration_img/cam{camera_id}/intrinsic/img*.png'):
    #     img = cv2.imread(filename)
    #     height, width, layers = img.shape
    #     size = (width,height)
    #     img_array.append(img)
    # out = cv2.VideoWriter(f'./calibration_img/cam{camera_id}/intrinsic/project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    