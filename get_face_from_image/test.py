import tensorflow as tf
import numpy as np
import cv2
import os
import detect_face

minsize = 20 
threshold = [ 0.6, 0.7, 0.7 ]  
factor = 0.709 


#restore mtcnn model
gpu_memory_fraction=1.0
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy/')
img=cv2.imread("test.jpg")
bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
for face_position in bounding_boxes:
	face_position=face_position.astype(int)
	cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
cv2.imshow("img",img)
cv2.waitKey(0)