import torch
def preprocess(dataset):
    path='../segmentation_dataset/'
	dataset1=np.load(os.path.join(path,'dataset_snr'+str(snr1)+'.npz'))
	data1=dataset1['data']
	label1=dataset1['label']
	dataset2=np.load(os.path.join(path,'dataset_snr'+str(snr2)+'.npz'))
	data2=dataset2['data']
	label2=dataset2['label']

	data1=np.expand_dims(data1, axis=4)
	label1=np.reshape(label1,(label1.shape[0],(np.prod(label1.shape[1:]))))
	# label1=keras.utils.to_categorical(label1,num_classes=2)

	data2=np.expand_dims(data2, axis=4)
	label2=np.reshape(label2,(label2.shape[0],(np.prod(label2.shape[1:]))))
	# label2=keras.utils.to_categorical(label2,num_classes=2)
	# Data preprocessing
	trainX=data1
	trainY=label1
	trainY=keras.utils.to_categorical(trainY,num_classes=2)

	testX=data2
	testY=label2

	testY=keras.utils.to_categorical(testY,num_classes=2)
	print(trainX.shape)
	print(trainY.shape)
	print(testX.shape)
	print(testY.shape)
	return trainX,trainY,testX,testY
