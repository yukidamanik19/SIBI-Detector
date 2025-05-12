import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.fingerTips = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        all_hands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myhand = {}
                ##lmlist = []
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## Bounding Box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                
                myhand['lmlist'] = mylmList
                myhand['bbox'] = bbox
                myhand['center'] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == 'Right':
                        myhand['type'] = 'Kiri'
                    else:
                        myhand['type'] = 'Kanan'
                else:
                    myhand['type'] = handType.classification[0].label
                all_hands.append(myhand)

                ## Draw Bounding Box
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), 
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myhand['type'], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 
                                2, (255, 0, 255), 2)
                print(myhand)
        if draw:
            return all_hands, img
        else:
            return all_hands
        
    def fingersUp(self, myHand):

        myHandType = myHand['type']
        mylmList = myHand['lmlist']
        if self.results.multi_hand_landmarks:
            fingers = []
            ## Thumb
            if myHandType == 'Kanan':
                if mylmList[self.tipIds[0]][0] > mylmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if mylmList[self.tipIds[0]][0] < mylmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
            ## 4 Fingers
            for id in range(1, 5):
                if mylmList[self.tipIds[id]][1] < mylmList[self.tipIds[id] - 2][1]:                        
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
            
    def findDistance(self, p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            return length, info, img
        else:
            return length, info

def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        ## Get Image Frame
        success, img = cap.read()
        ## Find the hands and its landmarks
        hands, img = detector.findHands(img)
        ## hands = detector.findHands(img, draw=False) ## without drawing

        if hands:
            ## hand 1
            hand1 = hands[0]
            lmlist1 = hand1['lmlist'] ## List of 21 landmarks
            bbox1 = hand1['bbox']
            centerPoint1 = hand1['center']
            handType1 = hand1['type']

            fingers1 = detector.fingersUp(hand1) ## [0, 1, 1, 0, 0]

            if len(hands) == 2:
                ## hand 2
                hand2 = hands[1]
                lmlist2 = hand2['lmlist']
                bbox2 = hand2['bbox']
                centerPoint2 = hand2['center']
                handType1 = hand2['type']

                fingers2 = detector.fingersUp(hand2)

                length, info, img = detector.findDistance(lmlist1[8][0:2], lmlist2[8][0:2], img)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
