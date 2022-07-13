from random import triangular
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import webbrowser
import urllib.request

import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

facial_angle_thresh = 15
ear_thresh = .20

face_counter=0
face_counter2 = 0
eye_counter =0
iris_counter = 0


total_distracted =0
framesIn5Sec = 150 #Initial Value
framesIn2Sec = 60
frameIn10Sec=300

sec5Counter= 0
offStates = [True,True,True,True]

isOff = True


def image_process():
    _, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image,results 


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def find_features(landmarks):
    face_3d = []
    face_2d = []
    leye = []
    liris = []
    bounding = []

    leyeval=[ 33, 160,158 , 133 ,153,144]
    headval = [ 33, 263, 1,  61, 291 , 199]
    lirisval = [469, 470, 471, 472]
    boundingval = [33,133]

    for i in lirisval: 
        x, y = face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height
        liris.append([x, y])  

    for i in boundingval: 
        x, y = face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height
        bounding.append([x, y])     

    for i in leyeval: 
        x, y = face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height
        leye.append([x, y])   

    for i in headval:
        x, y = face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height
        face_2d.append([x, y])
        face_3d.append([x, y, face_landmarks.landmark[i].z])    

    return face_3d,face_2d,leye,liris,bounding    

def eye_position(liris,bounding):
    arr = np.array(liris)

    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(arr.astype(np.int32))
    center_left = np.array([l_cx, l_cy], dtype=np.int32)

    total_distance =  dist.euclidean(bounding[0], bounding[1])
    #print(bounding[0])

    second_dist = dist.euclidean(center_left,bounding[0])
    ratio = second_dist/total_distance
    if ratio <= 0.42:
        centered = False
    elif ratio > 0.42 and ratio <= 0.57:
        centered = True
    else:
        centered =False

    return centered


def head_orientation(face_3d,face_2d,height,width):
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    cam_matrix = np.array([ [width, 0, height / 2], [0, width, width / 2],[0, 0, 1]])

    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    _, rotation_vector, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
    rotationmatrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _,_,_,_,_ = cv2.RQDecomp3x3(rotationmatrix)

    return angles

while cap.isOpened():
    start_time = time.time()

    image,results = image_process()
    height, width, _ = image.shape

    last_total_distracted = total_distracted

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]        

        mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
            
        face_3d,face_2d,leye,liris,bounding = find_features(face_landmarks)
        centered = eye_position(liris,bounding)

        for (x,y) in liris:
            cv2.circle(image, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)
        for (x,y) in bounding:
            cv2.circle(image, (int(x),int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        ratio = eye_aspect_ratio(leye)

        angles = head_orientation(face_3d,face_2d,height, width)
        yaw = angles[1] * 360
        pitch = angles[0] * 360


        prevStates=all(offStates)

        if (ratio < ear_thresh and eye_counter < framesIn2Sec):
            eye_counter += 1
        else:
            if(not(ratio < ear_thresh)):
                offStates[0] = True
            elif eye_counter >= framesIn2Sec:
                total_distracted += 1
                offStates[0] = False
            eye_counter = 0

        if ((yaw > facial_angle_thresh or yaw < -facial_angle_thresh)  and face_counter < framesIn5Sec):
            face_counter += 1
        else:
            if(not(yaw > facial_angle_thresh or yaw < -facial_angle_thresh)):
                offStates[1] = True
            elif face_counter >= framesIn5Sec:
                total_distracted += 1
                offStates[1] = False
            face_counter = 0

        if (centered != True and iris_counter < framesIn10Sec):
            iris_counter += 1
        else:
            if(not(centered != True)):
                offStates[2] = True
            elif iris_counter >= framesIn10Sec:
                total_distracted += 1
                offStates[2] = False
            iris_counter = 0

        if ((pitch > facial_angle_thresh or pitch < -facial_angle_thresh)  and face_counter2 < framesIn2Sec):
            face_counter2 += 1
        else:
            if(not(pitch > facial_angle_thresh or pitch < -facial_angle_thresh)):
                offStates[3] = True
            elif face_counter2 >= framesIn2Sec:
                total_distracted += 1
                offStates[3] = False
            face_counter2 = 0   

        totStates=all(offStates)


        if(totStates != prevStates):
            if isOff:
                print("On")
                isOff = False
            else:
                print("Off")
                isOff = True

        cv2.putText(image, "Distracted: {}".format(total_distracted), (15, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Yaw:: {}".format(int(yaw)), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Pitch:: {}".format(int(pitch)), (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "EAR: {:.2f}".format(ratio), (15, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Eyes Centered: {}".format(centered) , (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    


    FPS = 1.0 / (time.time() - start_time)
    framesIn5Sec = 5*(FPS)
    framesIn2Sec = 2*(FPS)
    framesIn10Sec = 10*(FPS)

    cv2.putText(image, "FPS: {:.2f}".format(FPS), (15, 230),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('YawEst', image)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows()
