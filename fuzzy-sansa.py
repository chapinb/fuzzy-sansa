__author__ = 'cbryce'
__license__ = 'Apache2'
__date__ = '20150409'
__version__ = '0.00'

"""
Fuzzy-sansa - an Open Source Facial Recognition Tool Maybe
Copyright 2015 Chapin Bryce

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import cv2


# Following guide from : https://realpython.com/blog/python/face-recognition-with-python/

def check_versions():
    if np.__version__ < "1.9.2" :
        return False
    if cv2.__version__ < "2.4.10":
        return False
    return True


def img(fin):

    casc = 'xml/haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(casc)

    # Read in image
    image = cv2.imread(fin)

    # Most libs work best in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect faces in image
    faces = faceCascade.detectMultiScale( # detectMultiScale is a general function to detect objects based on casc
        gray, # Hand gray image to process
        scaleFactor=1.1,  # Handles size of faces, since some may be closer/further
        minNeighbors=9,  # States minimum number of objects needed before the face is found
        minSize=(30,30),  # size of box to draw on face
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE  # Unsure....
    )

    print "Found {0} faces!".format(len(faces))

    for (x, y, w, h) in faces:
        """
        x: X position of rectangle
        y: Y position of rectangle
        w: Width of rectangle
        h: Height of rectangle
        """
        cv2.rectangle( # draws the rectangles around each face
            image,  # base image
            (x,y),  # starting coordinates
            (x+w, y+h),  # other 2 coordinates
            (0, 255, 0),  # maybe color? aka green if so
            2)  # no idea...maybe width of square?

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    cv2.imwrite(fin+'_out.jpg', image)

def vid(fin):
    video_capture = cv2.VideoCapture(0)
    casc = 'xml/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(casc)

    if not video_capture.isOpened():
        raise "Error opening video"

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def img_match(f1, f2):
    from matplotlib import pyplot as plt

    img1 = cv2.imread(f1, 0) # Query Image
    img2 = cv2.imread(f2, 0) # Training Image

    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)

    plt.imshow(img3), plt.show()


if __name__ == '__main__':
    if check_versions():
        import os
        # for r, d, f in os.walk('img'):
        #     for entry in f:
        #         if not entry.endswith('_out.jpg'):
        #             img(r+'/'+entry)
        # vid('mp4/1.mp4')
        img_match('img/7.png', 'img/4.png')