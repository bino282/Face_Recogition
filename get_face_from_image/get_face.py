import tensorflow as tf
import numpy as np
import cv2
import os
import detect_face

minsize = 100 
threshold = [ 0.6, 0.7, 0.7 ]  
factor = 0.709 


#restore mtcnn model
gpu_memory_fraction=1.0
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy/')

list_file=os.listdir("./image")
list_file=[f for f in list_file if f[-3:]]
index=0

for file in list_file:
	try:
		img=cv2.imread("./image/"+file)
		bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		nrof_faces = bounding_boxes.shape[0]
		for face_position in bounding_boxes:
			
			face_position=face_position.astype(int)
			
			crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
			crop=cv2.GaussianBlur(crop,(5,5),0)
			cv2.imwrite("./face_image/"+file,crop)
			index+=1
	except:
		continue