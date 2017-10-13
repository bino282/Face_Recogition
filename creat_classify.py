import tensorflow as tf
import numpy as np
import cv2
from numpy.linalg import norm
import os
import detect_face
import facenet
from sklearn.svm import SVC
import pickle
from sklearn.neighbors import KNeighborsClassifier

minsize = 100 
threshold = [ 0.6, 0.7, 0.7 ] 
factor = 0.709

model_dir='./pre_model'
path_model_classify="K_neighbors.pkl"

def load_data(path):
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
image_list,image_label,dict_map=load_data("./input_dir")	
list_feature=[]
with tf.Graph().as_default():
	with tf.Session() as sess:

		facenet.load_model(model_dir)
		images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
		embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
		phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			
		embedding_size = embeddings.get_shape()[1]
		print(embedding_size)
		for i in range (len(image_list)):
			imgi=image_list[i]
			imgi=norm_image(imgi)
			feature_imgi=sess.run([embeddings],feed_dict={images_placeholder: imgi,phase_train_placeholder: False })[0].flatten().tolist()
			list_feature.append(feature_imgi)
k_model=KNeighborsClassifier(n_neighbors=5)
k_model.fit(list_feature,image_label)
result=k_model.predict(list_feature)
count=0
for i in range(len(image_label)):
	if(result[i]==image_label[i]):
		count+=1
print ("Accuracy : "+str(count*1.0/len(image_label)))
model_name = 'K_neighbors.pkl'
with open(model_name,"wb") as f:
	pickle.dump(k_model,f)

