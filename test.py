import cv2, PIL, urllib, numpy as np, mediapipe as mp, matplotlib.pyplot as plt

mp_objectron = mp.solutions.objectron    # Tool to detect 3D objects
mp_drawing = mp.solutions.drawing_utils  # Tool to draw lines, boxes, axes on images

image_path = r"C:\Users\vamsi\Downloads\shoemy.jpg"
image = cv2.imread(image_path)
mug = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# step 2: LOAD THE OBJECT DETECTION MODEL
objectron = mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=0.2, model_name='Shoe')

# step 3: RUN THE DETECTION 
results = objectron.process(mug)

# step 4: CHECK IF OBJECT IS FOUND
if results.detected_objects:
    # step 5: DRAW THE OBJECT ON IMAGE
    annotated_image = mug.copy()  # copy of the original image

    for detected_object in results.detected_objects:
        mp_drawing.draw_landmarks(
            annotated_image,
            detected_object.landmarks_2d,
            mp_objectron.BOX_CONNECTIONS
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(annotated_image)
    ax.axis('off')  # useful for showing photos, not graphs
    plt.show()

else:
    print('No box landmarks detected.')