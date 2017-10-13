
import tensorflow as tf
import numpy as np
import cv2
from numpy.linalg import norm
import os
import detect_face
import facenet
import pickle
from sklearn.neighbors import KNeighborsClassifier
minsize = 100 
threshold = [ 0.6, 0.7, 0.7 ] 
factor = 0.709 

model_dir='./pre_model'
path_model_classify="K_neighbors.pkl"

def load_classify(filename):
	with open(filename,"rb") as f:
		model=pickle.load(f)
	return model
def load_data_test(path):
	dict_name={}
	dict_map={}
	index_name=0
	image_list=[]
	image_label=[]
	list_path_files=[]
	for subfolder in os.listdir(path):
		dict_name[subfolder]=index_name
		dict_map[index_name]=subfolder
		index_name+=1
		new_path=os.path.join(path,subfolder)
		files=os.listdir(new_path)
		for file in files:
			if(file.endswith(".jpg")):
				list_path_files.append(os.path.join(new_path,file))
				image_label.append(dict_name[subfolder])
	for idx,image_path in enumerate(list_path_files):
		img=cv2.imread(image_path)
		img=cv2.resize(img,(192,168))
		image_list.append(img)
	return image_list,image_label,dict_map
def norm_image(img):
	crop = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC )
	data=crop.reshape(-1,160,160,3)
	return data
def distance(a,b):
	return abs(np.dot(a, b)/(norm(a)*norm(b)))


print("load data input_dir ")
image_list,image_label,dict_map=load_data_test("./input_dir")
print ("load model classification ")
model=load_classify(path_model_classify)

#restore mtcnn model
print('Creating networks and loading parameters mtcnn')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy/')
list_feature=[]
with tf.Graph().as_default():
	with tf.Session() as sess:
		print( "Creating networks and loading parameters facenet ")	
		facenet.load_model(model_dir)
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
		embedding_size = embeddings.get_shape()[1]
		print(embedding_size)

		video_capture=cv2.VideoCapture(0)
		c=0
		while (True):
			try:
				find_results=[]
				ret,frame=video_capture.read()
				bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
				for face_position in bounding_boxes:
					face_position=face_position.astype(int)
					cv2.rectangle(frame, (face_position[0], 
							face_position[1]), 
					  (face_position[2], face_position[3]), 
					  (0, 255, 0), 2)
					crop=frame[face_position[1]:face_position[3],face_position[0]:face_position[2],]
					crop=cv2.GaussianBlur(crop,(5,5),0)
					crop=norm_image(crop)
					emb_img_test = sess.run([embeddings], feed_dict={images_placeholder: crop,phase_train_placeholder: False })[0].flatten().tolist()
					index=model.predict([emb_img_test])[0]
					find_results.append(dict_map[index])
				cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
				cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
				thickness = 2, lineType = 2)
				cv2.imshow('Video', frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			except:
				pass
		video_capture.release()
		cv2.destroyAllWindows()