# MEDIAPIPE VERSION 0.10.21
# ONLY 1 HAND (too fucking lazy to make it work for 2)
# only comparing x and y (again i am too lazy) i may or may not add z later
# this code will only work with something to process the serial communication 
# to work without remove all serial references

import cv2
import mediapipe as mp
import math
import serial

# linux reference hahahaha
port = serial.Serial("/dev/ttyACM0", 9600)

current = None
stable = 0
threshold = 10

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# returns a list of t/f for in order index, middle, ring, pinky
def bentFingers(landmarks):
    isBent = []

    # PIP, TIP, MCP
    for i, j, k in zip([6, 10, 14, 18], [8, 12, 16, 20], [5, 9, 13, 17]):
        # X, Y
        vA = [landmarks[k][0] - landmarks[i][0], landmarks[k][1] - landmarks[i][1]]
        vB = [landmarks[j][0] - landmarks[i][0], landmarks[j][1] - landmarks[i][1]]
            
        # dot product
        dotAB = (vA[0] * vB[0]) + (vA[1] * vB[1])

        # magnitude of each vector (length of bone segments)
        magnitudeA = math.sqrt((vA[0] * vA[0]) + (vA[1] * vA[1]))
        magnitudeB = math.sqrt((vB[0] * vB[0]) + (vB[1] * vB[1]))

        if magnitudeA == 0 or magnitudeB == 0:
            isBent.append(False)
            continue

        # compute cosine
        cos0 = dotAB / (magnitudeA * magnitudeB)

        # the finger is bent if cos0 is greater than -0.2
        if cos0 > -0.2:
            isBent.append(True)
        else:
            isBent.append(False)
    
    return isBent

def isThumbsUp(landmarks):
    bent = bentFingers(landmarks)
    return all(bent) and landmarks[4][1] < landmarks[2][1]
    
def isThumbsDown(landmarks):
    bent = bentFingers(landmarks)
    return all(bent) and landmarks[4][1] > landmarks[2][1]
    
def isMiddleFinger(landmarks):
    bent = bentFingers(landmarks)
    return bent == [True, False, True, True]

# main for webcam, landmarks etc
try:
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLMS in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(handLMS.landmark):

                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    
                    landmarks.append([cx, cy])
                    #print(landmarks)
                try:
                    #cv2.circle(img, (landmarks[4][0], landmarks[4][1]), 25, (255, 0, 255), cv2.FILLED)

                    gesture = None

                    if isThumbsUp(landmarks):
                        gesture = "THUMBS UP"
                    elif isThumbsDown(landmarks):
                        gesture = "THUMBS DOWN"
                    elif isMiddleFinger(landmarks):
                        gesture = "FUCK YOU"
                    
                    # check to not flood arduino
                    if gesture == current:
                        stable += 1
                    else:
                        current = gesture
                        stable = 0

                    if stable == threshold:
                        port.write((gesture + "\n").encode())
                        print(gesture)

                except Exception as e:
                    print(e)

                mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            port.write(("\n").encode())
            cap.release()
            cv2.destroyAllWindows()
            break

except KeyboardInterrupt:
    port.write(("\n").encode())
    cap.release()
    cv2.destroyAllWindows()