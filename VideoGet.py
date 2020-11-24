import cv2
import numpy as np
import tensorflow as tf
from threading import Thread

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.count = 0
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self, MIN_CONF_THRESH, detect_fn, category_index):
        Thread(target=self.get, args=(MIN_CONF_THRESH, detect_fn, category_index)).start()
        return self

    def get(self, MIN_CONF_THRESH, detect_fn, category_index):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                #ret, frame = videostream.read()
                frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                frame_expanded = np.expand_dims(frame_rgb, axis=0)
                imH, imW, _ = self.frame.shape

                # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                input_tensor = tf.convert_to_tensor(self.frame)
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
                self.count = 0
                for i in range(len(scores)):
                    if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
                        #increase count
                        self.count += 1
                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))

                        cv2.rectangle(self.frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                        # Draw label
                        object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
                        label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(self.frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(self.frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    def stop(self):
        self.stopped = True
