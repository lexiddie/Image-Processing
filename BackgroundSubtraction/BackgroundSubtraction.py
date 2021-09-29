import cv2

video_path = './NewYork.mov'

video_capture = cv2.VideoCapture(video_path)
rendering = cv2.createBackgroundSubtractorMOG2()

rendering_mask = None
success, frame = video_capture.read()

print(int(frame.shape[1]), int(frame.shape[0]))

while success:
    rendering_mask = rendering.apply(frame)
    count = 0
    success, frame = video_capture.read()
    cv2.imshow('New York', rendering_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.imwrite('background_subtraction.jpg', rendering_mask)
print(rendering_mask)
video_capture.release()
cv2.destroyAllWindows()
