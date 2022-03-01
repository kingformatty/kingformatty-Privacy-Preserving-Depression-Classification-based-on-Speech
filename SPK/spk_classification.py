import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
X = []
Y = []

with open('results_dict_list_baseline.pickle','rb') as f:
    results_dict_list = pickle.load(f)

for results_dict in results_dict_list:
    for i in range(len(results_dict['folder'])):
        X.append(results_dict['output'][i])
        Y.append(results_dict['folder'][i])
import pdb
#for embedding_file in embedding_files:
#     emb = pickle.load(open('IDL_embedding/'+embedding_file,'rb'))
#     X.append(emb)
#     Y.append(int(embedding_file.split('_')[0]))

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test  = train_test_split(X,Y,test_size=0.3, random_state=42)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train,y_train)

print(clf.score(X_test, y_test))


