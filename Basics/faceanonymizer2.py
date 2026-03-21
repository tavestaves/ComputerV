import cv2

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load cascade
face_cascade = cv2.CascadeClassifier('models/facess.xml')

if face_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()

# Main loop
while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #blur faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # Show frame
    cv2.imshow('frame', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()