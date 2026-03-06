Age & Gender Detection using AI
Project Overview

Age and Gender Detection is a computer vision project that predicts a person's age group and gender from a live camera feed or image. The system uses deep learning models with Convolutional Neural Networks (CNNs) to analyze facial features and classify them into predefined categories.

The application captures frames from a webcam, detects faces, and predicts the gender and age range for each detected face.

Technologies Used

Python

OpenCV (Computer Vision Library)

Deep Learning (CNN Models)

Pretrained Caffe Models

NumPy

Project Structure
Age-Gender-Detection
│
├── main.py
├── README.md
│
└── models
    ├── age_deploy.prototxt
    ├── age_net.caffemodel
    ├── gender_deploy.prototxt
    ├── gender_net.caffemodel
    ├── opencv_face_detector.pbtxt
    └── opencv_face_detector_uint8.pb
Features

Real-time face detection using webcam

Age group prediction

Gender classification

Deep learning model inference

Works on images or live video

Age Categories

The model predicts the following age ranges:

(0-2)
(4-6)
(8-12)
(15-20)
(25-32)
(38-43)
(48-53)
(60-100)
Installation
1 Install dependencies
pip install opencv-python numpy
2 Clone the repository
git clone https://github.com/yourname/Age-Gender-Detection.git
cd Age-Gender-Detection
3 Download pretrained models

Download the pretrained model files used for face detection and age/gender classification.

Example repository:

https://github.com/spmallick/learnopencv/tree/master/AgeGender

Place them inside the models folder.

Run the Project

Start the application using:

python main.py

The webcam will open and display predicted Age Range and Gender for each detected face.

Example output:

Male (25-32)
Female (15-20)
Applications

Age and gender detection systems are widely used in:

Smart advertising systems

Customer analytics

Security systems

Retail analytics

Human-computer interaction

Future Improvements

Improve accuracy using modern deep learning models

Use larger datasets for training

Deploy as a web application

Integrate with real-time analytics dashboards

Author

Avvaru Harikrishna

AI / Machine Learning Enthusiast
Interested in Computer Vision and Deep Learning.
