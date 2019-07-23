import numpy as np
import torch
import os
def sample_generation(pre,label,n):
    if not os.path.exists('samples'):
        os.mkdir('samples')
    for i in range(n):
        label_ = np.array(label[i].cpu())
        pre_ = np.array(pre[i].cpu())
        np.save('samples/label_%d.npy'%i,label_)
        np.save('samples/pre_%d.npy'%i,pre_)

    
def CORAL_3D(X_s,X_t,device):
    dim=np.prod(X_s.shape[1:])
    Cs = X_s.sum(0)
    Ct = X_t.sum(0)
    loss=torch.norm(Cs-Ct)/4/dim/dim
    return loss