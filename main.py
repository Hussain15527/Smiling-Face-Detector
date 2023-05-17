import cv2

# Load the Haar cascade classifier for detecting faces and smiles
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Set the width and height of the camera frame
cap.set(3, 640) # width
cap.set(4, 480) # height

# Loop over the frames from the camera
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face, detect smiles
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)

        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Check if the face is smiling and draw a rectangle around the mouth
        is_smiling = False
        for (sx,sy,sw,sh) in smiles:
            is_smiling = True
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

        # Draw a rectangle around the face with green if smiling, red if not
        if is_smiling:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    # Check if the user pressed the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
