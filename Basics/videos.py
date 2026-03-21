import os
import cv2

#read video

video_path = os.path.join ('.', 'assets', 'mevid.mp4')
video = cv2.VideoCapture(video_path)

#visualize video
ret = True
while ret:
    ret,frame = video.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):  # 33 ms for 30 FPS real-time playback, press 'q' to quit
            break