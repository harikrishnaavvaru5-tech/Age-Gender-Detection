import cv2
import numpy as np

# Model files
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']

genderList = ['Male', 'Female']


# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def detectFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0,
                                 (300, 300),
                                 [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(frameOpencvDnn,
                          (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

    return frameOpencvDnn, faceBoxes


video = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:

    hasFrame, frame = video.read()
    if not hasFrame:
        break

    resultImg, faceBoxes = detectFace(faceNet, frame)

    for faceBox in faceBoxes:

        face = frame[max(0, faceBox[1]-20):
                     min(faceBox[3]+20, frame.shape[0]-1),
                     max(0, faceBox[0]-20):
                     min(faceBox[2]+20, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0,
                                     (227, 227),
                                     MODEL_MEAN_VALUES,
                                     swapRB=False)

        # Gender Prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Age Prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"

        cv2.putText(resultImg, label,
                    (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

    cv2.imshow("Age Gender Detector", resultImg)

video.release()
cv2.destroyAllWindows()