#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Video) From TF2 Saved Model
=====================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from VideoGet import VideoGet
from CountsPerSec import CountsPerSec
#import RealTimeDetection

class ObjectDetection:

    def __init__(self):
        tf.get_logger().setLevel('ERROR')
        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load the model
    # ~~~~~~~~~~~~~~
    def load_custom_model(self, PATH_TO_MODEL_DIR, PATH_TO_LABELS):
        PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
        print('Loading model...', end='')
        start_time = time.time()

        # Load saved model and build the detection function
        global detect_fn
        detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        # Load label map data (for plotting)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        global category_index
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
        warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
        return detect_fn, category_index

    def putIterationsPerSec(self, frame, iterations_per_sec):
        """
        Add iterations per second text to lower-left corner of a frame.
        """

        cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
            (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        return frame


    def detectObjectSingleThread(self, MIN_CONF_THRESH):
        # Initialize Webcam
        videostream = cv2.VideoCapture(0)
        ret = videostream.set(3,1280)
        ret = videostream.set(4,720)
        cps = CountsPerSec().start()

        while True:
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = videostream.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)
            imH, imW, _ = frame.shape

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(frame)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # input_tensor = np.expand_dims(image_np, 0)
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                           for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


            # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            scores = detections['detection_scores']
            boxes = detections['detection_boxes']
            classes = detections['detection_classes']
            count = 0
            for i in range(len(scores)):
                if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
                    #increase count
                    count += 1
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    # Draw label
                    object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            frame = self.putIterationsPerSec(frame, cps.countsPerSec())
            cv2.putText (frame,'Objects Detected : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
            cv2.imshow('Objects Detector', frame)
            cps.increment()

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Done")


    def detectObjectMultiThread(self, MIN_CONF_THRESH):
        """
        Dedicated thread for grabbing video frames with VideoGet object.
        Main thread shows video frames.
        """
        video_getter = VideoGet(0).start(MIN_CONF_THRESH, detect_fn, category_index)
        cps = CountsPerSec().start()

        while True:
            if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
                video_getter.stop()
                break

            frame = video_getter.frame
            frame = self.putIterationsPerSec(frame, cps.countsPerSec())
            cv2.putText (frame,'Objects Detected : ' + str(video_getter.count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
            cv2.imshow("Objects Detector", frame)
            cps.increment()
        cv2.destroyAllWindows()
        print("Done")
