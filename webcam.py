import cv2


#read webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

#visualize webcam
while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()