#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser(description="average models")
    parser.add_argument("--exp_dir", required=True, type=str)
    parser.add_argument("--out_name", required=True, type=str)
    parser.add_argument("--num", default=12, type=int)
    parser.add_argument("--last_epoch", type=int)
    
    args = parser.parse_args()

    average_epochs = range(args.last_epoch - args.num, args.last_epoch, 1)
    #average_epochs = range(1,args.num+1)    
    print("average over", average_epochs)
    avg = None
    data_saver_avg = {}
    cnt = 0
    # sum
    path = os.path.join(args.exp_dir,'model','1')
    for epoch in average_epochs:
        import pdb
        file = os.path.join(path, 'md_'+str(epoch)+'_epochs.pth')
        ds_file = os.path.join(path,'data_saver_'+str(epoch)+'.pickle')
        checkpoint = torch.load(file, map_location=torch.device("cpu"))
        data_saver = pickle.load(open(ds_file,'rb'))
        if data_saver_avg == {}:
            data_saver_avg = data_saver
        else:
           for key in data_saver.keys():
                if key != 'class_weights':
                    data_saver_avg[key] += data_saver[key]
            #if file.endswith('data_saver.pickle'):
                #data_saver = pickle.load(open(os.path.join(path,file),'rb'))

                
        states = checkpoint['state_dict']
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
        #if cnt == 0:
        #    data_saver_avg['mean'] = data_saver['mean']
        #    data_saver_avg['std'] = data_saver['std']
        #else:
        #    data_saver_avg['mean'] += data_saver['mean']
        #    data_saver_avg['std'] += data_saver['std']
        #cnt += 1
    
    #data_saver_avg['pointer_one'] = data_saver['pointer_one']
    #data_saver_avg['pointer_zero'] = data_saver['pointer_zero']
    #data_saver_avg['index_ones'] = data_saver['index_ones']
    #data_saver_avg['index_zeros'] = data_saver['index_zeros']
    #data_saver_avg['temp_batch'] = data_saver['temp_batch']
    #data_saver_avg['class_weights'] = data_saver['class_weights']

    import pdb
    #pdb.set_trace()
    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = (avg[k] / args.num).type_as(avg[k])
    out_path = os.path.join(args.exp_dir, 'model','6')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(out_path,args.out_name)
    
    import pdb
    for k in data_saver_avg.keys():
        if k!= 'class_weights':
            data_saver_avg[k] = data_saver_avg[k] / args.num
    save_model = {}
    save_model['state_dict'] = avg
    save_model['epoch'] = checkpoint['epoch']
    save_model['optimizer'] = checkpoint['optimizer']
    save_model['rng_state'] = checkpoint['rng_state']
    save_model['cuda_rng_state'] = checkpoint['cuda_rng_state']
    save_model['numpy_rng_state'] = checkpoint['numpy_rng_state']
    save_model['random_rng_state'] = checkpoint['random_rng_state']
     
    torch.save(save_model, model_path)
    data_saver_path = os.path.join(out_path, 'data_saver.pickle')
    with open(data_saver_path,'wb') as f:
        pickle.dump(data_saver_avg,f)
    #data saver
    

if __name__ == "__main__":
    main()
