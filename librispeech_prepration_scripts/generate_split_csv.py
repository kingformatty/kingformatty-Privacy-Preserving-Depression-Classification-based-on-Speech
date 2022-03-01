import csv
import contextlib
import numpy as np
from sklearn.model_selection import train_test_split

folder_list =[]
# [['Participant_ID,PHQ8_Binary,PHQ8_Score,Gender,PHQ8_NoInterest,PHQ8_Depressed,PHQ8_Sleep,PHQ8_Tired,PHQ8_Appetite,PHQ8_Failure,PHQ8_Concentrating,PHQ8_Moving']]
#each element in folder_list has 12 entries
with open('folder_list.txt','r') as f:
    for line in f:
        elements = line.split('/')
        folder_name = elements[-1][:-3]
        #print(folder_name)
        folder_list.append([folder_name,folder_name,0,0,0,0,0,0,0,0,0,0])
        #print(folder_list)


#split 80% 10% 10%
import pdb
#pdb.set_trace()

train_index,dev_test_index = train_test_split(np.arange(len(folder_list)),test_size = 0.2, random_state = 42)

dev_index, test_index = train_test_split(dev_test_index,test_size = 0.5, random_state = 42)
train_index = list(train_index)
dev_index = list(dev_index)
test_index = list(test_index)
#pdb.set_trace()


print(train_index)
print(dev_index)
print(test_index)

#complete = [train_index]+[dev_index] + [test_index]

#complete_set = set(complete)
#print(len(complete_set))
import pdb
#pdb.set_trace()
#here we only generate trainset and devset, since testset is not necessary for test set

train_csv = '../labels/train_split_Depression_AVEC2017.csv'
with open(train_csv,mode='w') as csv_file:
    csv_file = csv.writer(csv_file,delimiter = ',')
    csv_file.writerow(['Participant_ID','PHQ8_Binary','PHQ8_Score','Gender','PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating,PHQ8_Moving'])
    for i in range(len(folder_list)):
        if i in train_index:
            content = folder_list[i]
            csv_file.writerow(content)
        #break

dev_csv = '../labels/dev_split_Depression_AVEC2017.csv'
with open(dev_csv,mode='w') as csv_file:
    csv_file = csv.writer(csv_file,delimiter=',')
    csv_file.writerow(['Participant_ID','PHQ8_Binary','PHQ8_Score','Gender','PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating,PHQ8_Moving'])
    for i in range(len(folder_list)):
        if i in dev_index:
            content = folder_list[i]
            csv_file.writerow(content)

full_train_csv = '../labels/full_train_split_Depression_AVEC2017.csv'
with open(full_train_csv,mode='w') as csv_file:
    csv_file = csv.writer(csv_file,delimiter = ',')
    csv_file.writerow(['Participant_ID','PHQ8_Binary','PHQ8_Score','Gender','PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating,PHQ8_Moving'])
    for i in range(len(folder_list)):
        if i in train_index or dev_index:
            content = folder_list[i]
            csv_file.writerow(content)

test_split_csv = '../labels/full_test_split.csv'
with open(test_split_csv,mode='w') as csv_file:
    csv_file = csv.writer(csv_file,delimiter = ',')
    csv_file.writerow(['Participant_ID','PHQ8_Binary','PHQ8_Score,Gender'])
    for i in range(len(folder_list)):
        if i in test_index:
            content = folder_list[i]
            csv_file.writerow([content[0],content[0],0,0])


complete_csv = '../labels/complete_Depression_AVEC2017.csv'
with open(complete_csv,mode='w') as csv_file:
    csv_file = csv.writer(csv_file,delimiter = ',')
    csv_file.writerow(['Participant_ID','PHQ8_Binary','PHQ8_Score','Gender','PHQ8_NoInterest','PHQ8_Depressed','PHQ8_Sleep','PHQ8_Tired','PHQ8_Appetite','PHQ8_Failure','PHQ8_Concentrating,PHQ8_Moving'])
    for i in range(len(folder_list)):
        content = folder_list[i]
        csv_file.writerow(content)

