## x-vector extraction

x-vector extraction is based on speechbrain toolbox, please see https://speechbrain.github.io/ for details.  
Please put x_vector_extraction.py script under speechbrain directory.
x-vector can be extracted by simply running python x_vector_extraction.py

## IDL Embedding Extraction

Configuration file setup is similar to Classification/DepAudioNet_reproduction. By running `python main_add_pretrain.py test --cuda --prediction_metric=1`, a `results_dict_list.pickle` and a folder with embedding for each speaker will be generated under current directory.  
Please modify the path in the code accordingly to meet your requirement. 

## Spk Classification

`spk_classification.py` will run a simply SVM classification task based on embedding generated in previous steps. Please change path of feature file accordingly. 
