import cv2
import os
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 11

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

class VideoCamera(object):
    def __init__(self, source=0, version=0):
        self.video = cv2.VideoCapture(source)  #If source is empty, it will default to live feed, else it will load a video file. This has been changed so that it can be used for both uploaded video and live feed - Mark
        self.version = version #Version refers to whether it will obscure or only detect, 0=detect, 1=obscure
        #ret = self.video.set(3,2000)
        #ret = self.video.set(4,2000)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        if image is None:
            print('Done')
            sys.exit()
            
        else:
            if self.version == 0: #Create detection overlay image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_expanded = np.expand_dims(image_rgb, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

                # Draw the results of the detection (aka 'visulaize the results')

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    min_score_thresh=0.60,
                    skip_scores=True)
            
                ret, jpeg = cv2.imencode('.jpg', image)
                #cv2.imshow(jpeg)
                #k = cv2.waitKey(200)
                return jpeg.tobytes() #Return detection overlay image
            else: #Create obscure image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_expanded = np.expand_dims(image_rgb, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

                # Draw the results of the detection (aka 'visulaize the results')

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=-1,
                    min_score_thresh=1,
                    skip_scores=True,
                    skip_labels=True)
                
                #Obscure image
                true_boxes = boxes[0][scores[0] > 0.95]    # true_boxes is always [] for some reason?
                if np.any(true_boxes):
                    height, width, channels = image.shape
                    ymin = true_boxes[0,0]*height
                    xmin = true_boxes[0,1]*width
                    ymax = true_boxes[0,2]*height
                    xmax = true_boxes[0,3]*width
                    B1=image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2].shape
                    #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),1]=np.random.randint(50, size=(B1[0],B1[1]))
                    image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2]=np.random.randint(10, size=(B1[0],B1[1]))
                
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
                
cv2.destroyAllWindows()
