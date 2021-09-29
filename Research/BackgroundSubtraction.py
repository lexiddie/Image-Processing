import cv2

video_path = './NewYork.mov'

video_capture = cv2.VideoCapture(video_path)
fps_count = video_capture.get(cv2.CAP_PROP_FPS)
# print('FPS', fps_count)
# print(video_capture.get(3), video_capture.get(4))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
rendering = cv2.createBackgroundSubtractorMOG2()
while video_capture.isOpened():
    success, frame = video_capture.read()
    # print(int(frame.shape[1]), int(frame.shape[0]))
    if success is False:
        break
    frame = cv2.resize(frame, (1280, 720))
    # rendering_mask = cv2.morphologyEx(rendering_mask, cv2.MORPH_OPEN, kernel)
    rendering_mask = rendering.apply(frame)
    cv2.imshow('Frame', rendering_mask)
    rendering_mask = cv2.flip(rendering_mask, 0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('before.jpg', frame)
        cv2.imwrite('after.jpg', rendering_mask)
        break

video_capture.release()
cv2.destroyAllWindows()
