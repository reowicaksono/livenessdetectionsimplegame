import sys
import argparse
import copy
import datetime
import pygame
from pygame.locals import *
import random

import numpy as np
import cv2 as cv

from facial_fer_model import FacialExpressionRecog

sys.path.append('../face_detection_yunet')
from yunet import YuNet

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install the latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]
parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--model', '-m', type=str, default='facial_expression_recognition_mobilefacenet_2022july.onnx',
                    help='Path to the facial expression recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pairs to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))

args = parser.parse_args()

def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()
    landmark_color = [
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255)  # left mouth corner
    ]

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)

        face_center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

        try:
            emoticon_path = "../facial_expression_recognition/emoticon/{}.png".format(fer_type)
            emoticon_image = cv.imread(emoticon_path)

            new_size = (bbox[2] // 2, bbox[3] // 2)
            emoticon_image = cv.resize(emoticon_image, new_size)

            mask = cv.threshold(emoticon_image[:, :, 2], 100, 255, cv.THRESH_BINARY)[1]
            emoticon_image[mask == 0] = [0, 0, 0]

            offset = 10
            emoticon_x = face_center[0] - emoticon_image.shape[1] // 2 + offset
            emoticon_y = face_center[1] - emoticon_image.shape[0] // 2

            output[emoticon_y:emoticon_y + emoticon_image.shape[0], emoticon_x:emoticon_x + emoticon_image.shape[1]] = emoticon_image
            cv.rectangle(output, (emoticon_x, emoticon_y), (emoticon_x + emoticon_image.shape[1], emoticon_y + emoticon_image.shape[0]), (0, 0, 255), 2)

        except Exception as e:
            print(f"Error loading emoticon for {fer_type}: {e}")

        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return fer_type, output

def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)
    for face_points in dets:
        fer_res = np.concatenate((fer_res, fer_model.infer(frame, face_points[:-1])), axis=0)
    return True, dets, fer_res


# Initialize the camera
cap = cv.VideoCapture(0)  # 0 corresponds to the default camera, you can change it if you have multiple cameras

# Initialize facial expression recognition model
fer_model = FacialExpressionRecog(args.model)


# shape parameters
size = width, height = (800, 800)
road_w = int(width/1.6)
roadmark_w = int(width/80)
# location parameters
right_lane = width/2 + road_w/4
left_lane = width/2 - road_w/4
# animation parameters
speed = 1

# initiallize the app
pygame.init()
running = True

# set window size
screen = pygame.display.set_mode(size)
# set window title
pygame.display.set_caption("Mariya's car game")
# set background colour
screen.fill((60, 220, 0))
# apply changes
pygame.display.update()

# load player vehicle
car = pygame.image.load("../car.png")
#resize image
#car = pygame.transform.scale(car, (250, 250))
car_loc = car.get_rect()
car_loc.center = right_lane, height*0.8

# load enemy vehicle
car2 = pygame.image.load("../otherCar.png")
car2_loc = car2.get_rect()
car2_loc.center = left_lane, height*0.2

detect_model = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2023mar.onnx')

counter = 0
speed = 5

while running:
    counter += 1

    # increase game difficulty overtime
    if counter == 5000:
        speed += 0.15
        counter = 0
        print("level up", speed)

    car2_loc[1] += speed
    if car2_loc[1] > height:
        if random.randint(0, 1) == 0:
            car2_loc.center = right_lane, -200
        else:
            car2_loc.center = left_lane, -200

    # end game logic
    if car_loc.colliderect(car2_loc):  # Use colliderect to check if the rectangles (cars) intersect
        print("GAME OVER! YOU LOST!")
        break

    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
        break

    # Perform facial expression recognition
    success, dets, fer_res = process(detect_model, fer_model, frame)

    if success:
        output_frame, ouputdata = visualize(frame, dets, fer_res)
        if "surprised" in output_frame and car_loc.centerx > left_lane:
            car_loc = car_loc.move([-int(road_w / 2), 0])
            print("ini ke sini")
        elif "happy" in output_frame and car_loc.centerx < right_lane:
            car_loc = car_loc.move([int(road_w / 2), 0])
            print("ini ke senang")

        # Display the frame using Pygame
        pygame_frame = cv.cvtColor(ouputdata, cv.COLOR_BGR2RGB)
        pygame_surface = pygame.surfarray.make_surface(pygame_frame)

        # Flip the camera feed horizontally for correct orientation
        pygame_surface = pygame.transform.flip(pygame_surface, True, False)

        # Create two separate surfaces for the road and the camera feed
        road_surface = pygame.Surface((width, height))
        road_surface.fill((50, 50, 50))

        # Draw road markings on the road surface
        pygame.draw.rect(road_surface, (255, 240, 60), (width / 2 - roadmark_w / 2, 0, roadmark_w, height))
        pygame.draw.rect(road_surface, (255, 255, 255),
                         (width / 2 - road_w / 2 + roadmark_w * 2, 0, roadmark_w, height))
        pygame.draw.rect(road_surface, (255, 255, 255),
                         (width / 2 + road_w / 2 - roadmark_w * 3, 0, roadmark_w, height))

        # Blit the road and camera feed surfaces onto the main screen
        screen.blit(road_surface, (0, 0))
        screen.blit(pygame_surface, (width // 2, 0))

        # Place car images on the screen
        screen.blit(car, car_loc)
        screen.blit(car2, car2_loc)

        # Apply changes
        pygame.display.update()

    # Event handling and other logic

# Release the camera outside the loop
cap.release()

# Collapse application window
pygame.quit()