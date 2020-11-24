
import argparse
from ObjectDetection import ObjectDetection

class RealTimeDetection:

        def get_user_input(self):
            #tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                                default='exported-models/my_mobilenet_model')
            parser.add_argument('--labels', help='Where the Labelmap is Located',
                                default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
            parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                                default=0.5)
            parser.add_argument('--multithread', help='Specify whether to execute as multithreaded application or not. Default is False',
                                default='False')
            args = parser.parse_args()
            # PROVIDE PATH TO MODEL DIRECTORY
            PATH_TO_MODEL_DIR = args.model
            # PROVIDE PATH TO LABEL MAP
            PATH_TO_LABELS = args.labels
            # PROVIDE THE MINIMUM CONFIDENCE THRESH OLD
            MIN_CONF_THRESH = float(args.threshold)
            # PROVIDE SINGLE OR MULTI THREADED
            MULTI_THREADED = args.multithread
            return  PATH_TO_MODEL_DIR, PATH_TO_LABELS, MIN_CONF_THRESH, MULTI_THREADED

if __name__ == '__main__':
    modelPath, labelPath, threshold, multithread = RealTimeDetection().get_user_input()
    objDet = ObjectDetection()
    detect_fn, category_index = objDet.load_custom_model(modelPath, labelPath)

    if multithread == 'True':
        print("multithread")
        objDet.detectObjectMultiThread(threshold)
    else:
        print("single thread")
        objDet.detectObjectSingleThread(threshold)
