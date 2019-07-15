import torch
import torch.nn as nn
import os
import numpy as np
import keras
import argparse
from torch.utils.data import DataLoader
from CECT_dataloader import CECT_dataset
from model import *
from tqdm import tqdm
def meanIoU(y_pred, y_true):
    iou=np.zeros(2)
    y_pred = np.argmax(y_pred,axis=-1).astype(bool)
    y_true = np.argmax(y_true,axis=-1).astype(bool)

    al=y_pred.shape[1]
    pos=np.sum(y_pred*y_true,axis=1)
    neg=np.sum((~y_pred)*(~y_true),axis=1)
    # pos=float(np.sum(y_pred*y_true))
    # neg=float(np.sum((~y_pred)*(~y_true)))
    
    iou[0]=np.mean(neg/(al-pos))
    iou[1]=np.mean(pos/(al-neg))
    
    return np.mean(iou)
# if the dataloader_snr is not created, we can use this function to build dataloader_snr
def preprocess(snr):
	path='../segmentation_dataset/'
	dataset=np.load(os.path.join(path,'dataset_snr'+str(snr)+'.npz'))
	data=dataset['data']
	label=dataset['label']

	data=np.expand_dims(data, axis=4)
	label1=np.reshape(label,(label.shape[0],(np.prod(label.shape[1:]))))
	label=keras.utils.to_categorical(label1,num_classes=2)
	X=data
	Y=label1
	Y=keras.utils.to_categorical(Y,num_classes=2)
	X=np.array(X)
	Y=np.array(Y)
	dataloader_path = os.path.join(path,'dataloader_%d'%(snr))
	if not os.path.exists(dataloader_path):
		os.mkdir(dataloader_path)
		for i in range(len(X)):
			data_path = os.path.join(dataloader_path,'%d.npz'%i)
			np.savez(data_path,data=X[i],label=Y[i])
	else:
		print('The dataloader directory is created previously, therefore this function can be discarded.')

def main(opt):
	use_cuda=True if torch.cuda.is_available() else False
	device=torch.device('cuda:2') if use_cuda else torch.device('cpu')
	device_ids = [2]
	torch.manual_seed(1)
	if use_cuda:
		torch.cuda.manual_seed(1)
	cectA_dataset = CECT_dataset(path=opt['src_data'])
	cectA_dataloader = DataLoader(dataset=cectA_dataset,batch_size=opt['batch_size'],shuffle=True)
	cectB_dataset = CECT_dataset(path=opt['tar_data'])
	cectB_dataloader = DataLoader(dataset=cectB_dataset,batch_size=opt['batch_size'],shuffle=True)
	modelA = FCN_nd(aux_output=True)
	modelA = modelA.to(device)

	modelB = FCN_nd(aux_output=True)
	modelB = modelB.to(device)

	discriminator = Discriminator()
	discriminator = discriminator.to(device)
	# modelA = nn.DataParallel(modelA, device_ids=device_ids)
	# modelB = nn.DataParallel(modelB, device_ids=device_ids)
	# discriminator = nn.DataParallel(discriminator, device_ids=device_ids)
##########################################################################
	optimizer=torch.optim.Adam(modelA.parameters(),lr=0.001,betas=(0.9,0.99), eps=0, weight_decay=1e-8)

	for _ in range(opt['n_epoches']):
			total_loss = 0
			print('training procedure......')
			for data,label in tqdm(cectA_dataloader):
				data = data.to(device)
				label = label.to(device)
				# label = torch.reshape(label,(np.prod(label.shape[0:2]),)).to(device)
				pred,_ = modelA(data)
				# print(list(model.named_parameters()))
				optimizer.zero_grad()
				criteria=torch.nn.BCELoss()
				loss = criteria(pred,label)
				loss.backward()
				total_loss += loss
				optimizer.step()
			print('testing procedure......')
			with torch.no_grad():
				prediction = ()
				testY = ()
				for data,label in tqdm(cectB_dataloader):
					data = data.to(device)
					label = label.to(device)
					pred,_ = modelA(data)
					prediction += (np.array(pred.cpu()),)
					testY += (np.array(label.cpu()),)
				prediction = np.concatenate(prediction,axis=0)
				testY = np.concatenate(testY,axis=0)
				mIoU=meanIoU(prediction, testY)
				print('mIoU = %f'%mIoU)
##########################################################################
	modelB.load_state_dict(modelA.state_dict())
	optimizer_D=torch.optim.Adam(discriminator.parameters(),lr=0.001,betas=(0.9,0.99), eps=0, weight_decay=1e-8)
	optimizer_F=torch.optim.Adam(modelB.parameters(),lr=0.001,betas=(0.9,0.99), eps=0, weight_decay=1e-8)
	for _ in range(opt['n_epoches']):
		total_loss = 0
		print('training procedure......')
		for packA,packB in tqdm(zip(cectA_dataloader,cectB_dataloader)):
			dataA, labelA = packA
			dataA = dataA.to(device)
			labelA = labelA.to(device)
			dataB, _ = packB
			dataB = dataB.to(device)
			predA, auxA = modelA(dataA)
			_, auxB = modelB(dataB)
			
			aux = torch.cat([auxA,auxB],dim=0)

			domain_label = torch.cat([torch.zeros(opt['batch_size']),torch.ones(opt['batch_size'])],dim=0).to(device)
			domain_label = domain_label.long()
			domain_pred = discriminator(aux)
			if domain_label.shape[0] != domain_pred.shape[0]:
				# print(domain_label.shape[0],domain_pred.shape[0])
				continue
			# print(list(model.named_parameters()))
			optimizer_F.zero_grad()
			optimizer_D.zero_grad()
			criteria_D=torch.nn.CrossEntropyLoss()
			loss_D = criteria_D(domain_pred,domain_label)
			loss_D.backward()
			optimizer_F.step()
			optimizer_D.step()
		print('testing procedure......')
		with torch.no_grad():
			prediction = ()
			testY = ()
			for data,label in tqdm(cectB_dataloader):
				data = data.to(device)
				label = label.to(device)
				pred,_ = modelB(data)
				prediction += (np.array(pred.cpu()),)				
				testY += (np.array(label.cpu()),)
			prediction = np.concatenate(prediction,axis=0)
			testY = np.concatenate(testY,axis=0)
			mIoU=meanIoU(prediction, testY)
			print('mIoU = %f'%mIoU)

if __name__ == '__main__':
	preprocess(500)
	parser=argparse.ArgumentParser()
	parser.add_argument('--n_epoches',type=int,default=2)
	parser.add_argument('--batch_size',type=int,default=20)
	parser.add_argument('--src_data',type=str,default='../segmentation_dataset/dataloader_10000')
	parser.add_argument('--tar_data',type=str,default='../segmentation_dataset/dataloader_500')
	opt=vars(parser.parse_args())
	main(opt)

