import cv2, mediapipe as mp, matplotlib.pyplot as plt

mp_objectron = mp.solutions.objectron    # Tool to detect 3D objects
mp_drawing = mp.solutions.drawing_utils  # Tool to draw lines, boxes, axes on images

mug = cv2.VideoCapture(r'C:\Users\vamsi\Downloads\shoe_animation.mp4')

# step 2: LOAD THE OBJECT DETECTION MODEL
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.2, min_tracking_confidence=0.7, model_name='Shoe')

# step 3: LOAD THROUGH EACH FRAME OF THE VIDEO
while mug.isOpened():
    success, image = mug.read() # success is true frame is read correctly
    if not success:
        break

    # step 4: PREPROCESS IMAGE (REQUIRED FOR MEDIAPIPELINE)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # step 5: RUN THE DETECTION 
    results = objectron.process(image)

    # step 6: DRAW THE 3D BOX AND AXIS
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert back for OpenCv to display it

    # step 7: CHECK IF OBJECT IS FOUND
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )

            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

    # step 8: DISPLAY OUTPUT
    cv2.imshow('Mediapipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

mug.release()
cv2.destroyAllWindows()