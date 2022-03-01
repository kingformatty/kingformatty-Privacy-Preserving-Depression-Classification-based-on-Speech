import wave
import contextlib
import csv
#read folder name
folder_list = []
with open('folder_list.txt','r') as f:
    for line in f:
        folder_list.append(line[:-1])

#read file and save duration

for folder in folder_list:
    elements = folder.split('/')
    fname = '../'+folder+'/'+elements[-1][:-2]+'_AUDIO.wav'
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(rate)
        print(duration)
    
    #save results in csv file
    #with the form 
    ###################
    #start_time stop_time speaker value (speaker value is fixed to be 'Partcipant')
    transcript_csv_dir = '../'+folder+'/'+elements[-1][:-2]+'_TRANSCRIPT.csv'
    with open(transcript_csv_dir,mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter = '\t')
        csv_file.writerow(['start_time','stop_time','speaker','value'])
        csv_file.writerow([0.00,duration,'Participant','None'])
    #break


#print(folder_list)
