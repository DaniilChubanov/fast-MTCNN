import tensorflow as tf
from align import detect_face
import cv2
import imutils
import argparse
import time
import os
import ctypes
from threading import Thread, Lock
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "0" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "20" # export NUMEXPR_NUM_THREADS=6
#parser = argparse.ArgumentParser()
#parser.add_argument("--img", type = str, required=True)
#args = parser.parse_args()
import numpy as np



# some constants kept as default from facenet
minsize = 40
threshold = [0.55, 0.65, 0.70]
factor = 0.7098
margin = 0
input_image_size = 160

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(
    sess, '/Users/amir/person_recognition/src/align')


def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    print (bbox_resizer(bounding_boxes,img))
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.50:
                #det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                _margin = float(np.divide(margin, 2))
                bb[0] = np.maximum(np.subtract(face[0], _margin), -float(img_size[1]))
                bb[1] = np.maximum(np.subtract(face[1], _margin), -float(img_size[0]))
                bb[2] = np.minimum(np.add(face[2], _margin),
                                   float(img_size[1]))
                bb[3] = np.minimum(np.add(face[3], _margin),
                                   float(img_size[0]))
                cropped = img[bb[1]: bb[3], bb[0]: bb[2], :]
                #print (cropped.shape)
                #resized = cv2.resize(cropped, dsize=(input_image_size, input_image_size))
                #cv2.imshow("resize", cropped)
                # print(resized.shape)
                faces.append({'face': cropped, 'rect': [
                             bb[0], bb[1], bb[2], bb[3]], 'acc': face[4]})
                #print (faces)
    return faces

def bbox_resizer(bboxs,size_frame=(200,200)):
    # imageToPredict = cv2.imread("img.jpg", 3)
    # Note: flipped comparing to your original code!
    # x_ = imageToPredict.shape[0]
    # y_ = imageToPredict.shape[1]
    _bbox=[]
    for bbox in bboxs:
        y_ = frame.shape[0]
        x_ = frame.shape[1]
        x_scale = np.divide(size_frame[0] , x_)
        y_scale = np.divide(size_frame[1] , y_)
        #print(x_scale, y_scale)
        #frame = cv2.resize(frame, resize);
        #print(img.shape)
        #img = np.array(img);

        # original frame as named values
        print(bbox.shape)
        (origLeft, origTop, origRight, origBottom) = bbox.shape[0:4]

        x = int(np.round(np.multiply(origLeft , x_scale)))
        y = int(np.round(np.multiply(origTop , y_scale)))
        xmax = int(np.round(np.multiply(origRight , x_scale)))
        ymax = int(np.round(np.multiply(origBottom, y_scale)))
        _bbox.append([x,y,xmax,ymax])
    return {}
    # Box.drawBox([[1, 0, x, y, xmax, ymax]], img)
    #drawBox([[1, 0, x, y, xmax, ymax]], img)

class WebcamVideoStream:
    def __init__(self, src=0, width=500, height=500, fps=30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                frame = np.array(frame)
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


if __name__ == "__main__":
    vs = WebcamVideoStream(src=0, width=200, height=200, fps=60).start()
    while True:
        frame = vs.read()
        # if ret:
        #frame = np.array(frame)
    #img = imutils.resize(img, width=1000)
        timestart = time.clock()
        faces = getFace(frame)
        print(time.clock()-timestart)
        for face in faces:
            cv2.rectangle(frame, (face['rect'][0], face['rect'][1]),
                          (face['rect'][2], face['rect'][3]), (0, 255, 0), 2)
        cv2.imshow("faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()
