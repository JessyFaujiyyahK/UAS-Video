import cv2
import numpy as np

video = cv2.VideoCapture("green screen.mp4")
image = cv2.imread('bg.jpg')

# Mendefinisikan fungsi kompresi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 300, 300)

cv2.createTrackbar('L - H', "Trackbars", 0, 179, nothing)
cv2.createTrackbar('L - S', "Trackbars", 0, 255, nothing)
cv2.createTrackbar('L - V', "Trackbars", 0, 255, nothing)
cv2.createTrackbar('U - H', "Trackbars", 179, 179, nothing)
cv2.createTrackbar('U - S', "Trackbars", 255, 255, nothing)
cv2.createTrackbar('U - V', "Trackbars", 255, 255, nothing)
 
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    image = cv2.resize(image, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos('L - H', "Trackbars")
    l_s = cv2.getTrackbarPos('L - S', "Trackbars")
    l_v = cv2.getTrackbarPos('L - V', "Trackbars")
    u_h = cv2.getTrackbarPos('U - H', "Trackbars")
    u_s = cv2.getTrackbarPos('U - S', "Trackbars")
    u_v = cv2.getTrackbarPos('U - V', "Trackbars")

    l_green = np.array([l_h, l_s, l_v])
    u_green = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame - res
    green_screen = np.where(f == 0, image, f)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    output.write(green_screen)  # Menulis frame ke video output
    cv2.imshow("green_screen", green_screen)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
output.release()  # Melepaskan video output
cv2.destroyAllWindows()