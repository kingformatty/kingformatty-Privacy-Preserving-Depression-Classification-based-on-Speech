import os
import numpy as np
import pickle
import cv2


IDL_emb_file = os.listdir('Baseline_embedding')
x_vec_emb_file = os.listdir('embedding')
corr_list = []
for IDL_emb in IDL_emb_file:
    IDL = pickle.load(open('Baseline_embedding/'+IDL_emb,'rb'))
    spk_id = IDL_emb.split('_')[0]
    max_corr = 0
    x_vec_emb = spk_id+'_emb.pickle'
    #for x_vec_emb in x_vec_emb_file:
    if True:
        x_vec = pickle.load(open('embedding/'+x_vec_emb,'rb'))
        #calculate correlation
        x_vec = x_vec[0,0,:].numpy()
        if len(x_vec)!=len(IDL):
        #resize x_vec
            import pdb
            x_vec = cv2.resize(x_vec,(1,len(IDL)))[:,0]
        mat = np.corrcoef(x_vec,IDL)
        if mat[0,1]>=max_corr:
           max_corr = mat[0,1]
         
    corr_list.append(max_corr)
print('Average correlation is')
print(np.mean(corr_list))
