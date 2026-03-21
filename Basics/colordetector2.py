# detect hand and color of object in hand 
import cv2

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load cascade
fist_cascade = cv2.CascadeClassifier('models/fist.xml')
left_cascade = cv2.CascadeClassifier('models/left.xml')
lpalm_cascade = cv2.CascadeClassifier('models/lpalm.xml')
right_cascade = cv2.CascadeClassifier('models/right.xml')
rpalm_cascade = cv2.CascadeClassifier('models/rpalm.xml')

if fist_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()
if left_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()
if lpalm_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()
if right_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()
if rpalm_cascade.empty():
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

    # Detect hands
    fists = fist_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    lefts = left_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    lpalms = lpalm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    rights = right_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    rpalms = rpalm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

    # Draw rectangles
    for (x, y, w, h) in fists:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in lefts:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in lpalms:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in rights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in rpalms:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow('frame', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()