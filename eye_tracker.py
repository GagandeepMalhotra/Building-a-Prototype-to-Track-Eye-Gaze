import cv2
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras import layers, models

#Define mediapipe library variables
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh

#Left and Right irises indices
LEFT_IRIS_ID = [474,475, 476, 477]
RIGHT_IRIS_ID = [469, 470, 471, 472]

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  

FACE_CENTER_ID = [168]

#Reads the camera output
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(3, 64)
cap.set(4, 48)
model = models.load_model('model/')

def main(averaged_look_vector, selected_display):
    with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
                               min_detection_confidence=0.5,min_tracking_confidence=0.99 
    ) as face_mesh:
        results_detection, results_mesh, frame = read_frame(cap, face_mesh)
        left_cropped_eye_region = None
        right_cropped_eye_region = None
        reRatio = 0
        leRatio = 0
        if results_detection.detections is not None:
            for detection in results_detection.detections:
                #Get left and right eye keypoints
                left_eye = get_eye_keypoints("LEFT_EYE", frame, detection, mp_face_detection)
                right_eye = get_eye_keypoints("RIGHT_EYE", frame, detection, mp_face_detection)

                #Get size of both eyes
                eye_width, eye_height = get_eyes_dimensions(left_eye, right_eye)

                #Get left and right eye regions
                left_eye_region = get_eye_region(left_eye, eye_width, eye_height, frame)
                right_eye_region = get_eye_region(right_eye, eye_width, eye_height, frame)


                #Process left and right eye region by applying relevant functions
                if left_eye_region is not None and left_eye_region.shape[0] > 0 and left_eye_region.shape[1] > 0 and right_eye_region is not None and right_eye_region.shape[0] > 0 and right_eye_region.shape[1] > 0:
                    if results_mesh.multi_face_landmarks:
                        left_look_vector, left_cropped_eye_region, left_pupil_center = process_eye_region(left_eye_region, LEFT_IRIS_ID, results_mesh, frame)
                        right_look_vector, right_cropped_eye_region, right_pupil_center = process_eye_region(right_eye_region, RIGHT_IRIS_ID, results_mesh, frame)
                        averaged_look_vector = [(x + y) / 2 for x, y in zip(left_look_vector, right_look_vector)]
                        left_cropped_eye_region = cv2.flip(left_cropped_eye_region, 1)
                        right_cropped_eye_region = cv2.flip(right_cropped_eye_region, 1)
                        if selected_display == 1:
                            left_arrow_end = get_gaze_values(left_pupil_center, averaged_look_vector, 50)
                            right_arrow_end = get_gaze_values(right_pupil_center, averaged_look_vector, 50)
                            plot_gaze(left_pupil_center, left_arrow_end, frame)
                            plot_gaze(right_pupil_center, right_arrow_end, frame)

                        elif selected_display == 2:
                            face_center = get_face_center(frame, results_mesh)
                            face_center_arrow_end = get_gaze_values(face_center, averaged_look_vector, 100)
                            plot_gaze(face_center, face_center_arrow_end, frame)
                        
                        else:
                            left_arrow_end = get_gaze_values(left_pupil_center, averaged_look_vector, 25)
                            right_arrow_end = get_gaze_values(right_pupil_center, averaged_look_vector, 25)
                            plot_gaze(left_pupil_center, left_arrow_end, frame)
                            plot_gaze(right_pupil_center, right_arrow_end, frame)

                            face_center = get_face_center(frame, results_mesh)
                            face_center_arrow_end = get_gaze_values(face_center, averaged_look_vector, 100)
                            plot_gaze(face_center, face_center_arrow_end, frame)

                        #display_window(left_cropped_eye_region, "Left Eye Window")
                        #display_window(right_cropped_eye_region, "Right Eye Window")
                        #display_window(frame, 'Full Webcam')

                        mesh_coords = landmarksDetection(frame, results_mesh, False)
                        reRatio, leRatio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                        
        frame = cv2.flip(frame, 1)

        return frame, averaged_look_vector, left_cropped_eye_region, right_cropped_eye_region, reRatio, leRatio

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    #list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

    #returning the list of tuples for each landmarks 
    return mesh_coord

#Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

#Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    #Right Eyes 
    #horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    #Left Eyes
    #horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    #vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]
    
    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    if rvDistance != 0:
        reRatio = rhDistance/rvDistance
    else:
        reRatio = 0

    if lvDistance != 0:
        leRatio = lhDistance/lvDistance
    else:
        leRatio = 0

    return reRatio, leRatio

def get_face_center(frame, results_mesh):
    p = results_mesh.multi_face_landmarks[0].landmark[168]
    face_center = np.array([p.x * frame.shape[1], p.y * frame.shape[0]], dtype=int)
    return face_center 

#Gets the positions of keypoints of the eye
def get_eye_keypoints(chosen_eye, frame, detection, mp_face_detection):
    keypoints = detection.location_data.relative_keypoints
    eye_keypoint = getattr(mp_face_detection.FaceKeyPoint, chosen_eye)
    eye = (int(keypoints[eye_keypoint].x * frame.shape[1]),
            int(keypoints[eye_keypoint].y * frame.shape[0]))
    return eye

#Gets the width and height of the eye
def get_eyes_dimensions(left_eye, right_eye):
    eye_width = int(abs(left_eye[0] - right_eye[0]) * 1)
    eye_height = int(eye_width * 0.75)
    return eye_width, eye_height

#Gets the position of the eye in the frame
def get_eye_region(eye, eye_width, eye_height, frame):
    eye_x = int(eye[0] - eye_width * 0.5)
    eye_y = int(eye[1] - eye_height * 0.65)

    eye_region = frame[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width]
    return eye_region

def get_pupil_center(chosen_iris, frame, results_mesh):
    #Stores all x and y coordinates of facial landmarks
    mesh_points = np.array([np.multiply([p.x, p.y], frame.shape[:2][::-1]).astype(int) 
                            for p in results_mesh.multi_face_landmarks[0].landmark])
    #Define and plot the pupil center position
    (cx, cy), radius = cv2.minEnclosingCircle(mesh_points[chosen_iris])
    pupil_center = np.array([cx, cy], dtype=np.int32)
    return pupil_center

#Define the look vector
def get_look_vector_prediction(eye_pic_data):
    prediction = model.predict(eye_pic_data, verbose=0)
    look_vector = prediction[0]
    clipped_look_vector = [max(-1, min(1, x)) for x in look_vector]
    return clipped_look_vector

def read_window(eye_region):
    cropped_eye_region = cv2.resize(eye_region, (120, 80))
    eye_pic_data = read_image(cropped_eye_region)
    return eye_pic_data, cropped_eye_region

def display_window(eye_region, window_name):
    cv2.imshow(window_name, eye_region)

def read_image(image):
    pic_rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pic_rgb_arr = pic_rgb_arr.reshape((1, 80, 120, 3))
    return pic_rgb_arr

def get_gaze_values(pupil_center, look_vector, length):
    #Convert look vector to pitch and yaw
    pitch = math.asin(look_vector[1])
    yaw = math.atan2(look_vector[0], -look_vector[2])

    #Define and apply ascaling factor based on the z coordinate and arrow length
    dz = length * math.cos(pitch) * math.cos(yaw)
    z_scale = max(0.1, dz / length)
    length *= z_scale

    #Convert pitch and yaw angles to position
    dx = length * math.cos(pitch) * math.sin(yaw)
    dy = length * math.sin(pitch)
    dz = length * math.cos(pitch) * math.cos(yaw)

    #Get end point in relation to pupil center
    end_x = int(pupil_center[0] + dx)
    end_y = int(pupil_center[1] + dy)
    #end_z = int(dz)

    return end_x, end_y

def plot_gaze(pupil_center, arrow_end, frame):
    #Plot the pupil center
    cv2.circle(frame, pupil_center, 1, (0,255,0), 2, cv2.LINE_AA)
    #Plot the gaze line
    cv2.arrowedLine(frame, pupil_center, arrow_end[:2], (0, 0, 255), thickness=2)

#Exit program if 'q' is pressed
def exit_program(cap):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

def read_frame(cap, face_mesh):
    #Read frame and convert to RGB
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Define the face detection and mesh
    results_detection = face_detection.process(frame_rgb)
    results_mesh = face_mesh.process(frame_rgb)
    return results_detection, results_mesh, frame

#Calls all functions with associated eye values if eye_region values are valid
def process_eye_region(eye_region, iris_id, results_mesh, frame):
    eye_pic_data, cropped_eye_region = read_window(eye_region)
    look_vector = get_look_vector_prediction(eye_pic_data)
    #print("look vector: ", look_vector)
    pupil_center = get_pupil_center(iris_id, frame, results_mesh)

    return look_vector, cropped_eye_region, pupil_center

"""
#Calls the main function
if __name__ == '__main__':
    look_vector = [0,0,0]
    main(look_vector)
"""
