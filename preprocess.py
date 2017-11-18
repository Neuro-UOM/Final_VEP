import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def preprocess(label, fileName):
    
    data = np.loadtxt(fileName , usecols=range(7,19),delimiter=',' , skiprows=1)
    data = data[:,[0,2,4,6,8,10]]
    
    arr = []
    
    for i in range(0,6):
        y = data[:,i]
        y = butter_highpass_filter(y,5,132,5)
        ps = np.abs(np.fft.fft(y))**2
        arr.append(ps)
        
    freqs = np.fft.fftfreq( data[:,0].size , float(1)/128 )
    arr.append(freqs)
    
    output_arr = np.array(arr)
    output_arr = np.transpose(output_arr)    
    
    # DROP WHEN FREQUENCY < 0
    output_arr = output_arr[ np.logical_not( output_arr[:,6] < 0 ) ]
    
    # Add label column
    num_rows, num_cols = output_arr.shape
    label_column = np.full((num_rows, 1), label)  
    output_arr = np.hstack(( output_arr , label_column ))
    
    ### *********************************************************************
    for i in range(0,6):
        output_arr = output_arr[ np.logical_not( output_arr[:,i] < 100000 ) ]
    ###
    
    print output_arr.shape    
    return output_arr
    


blue_csv =  ['data/raw_hansika_blue.csv',   'data/raw_heshan_blue.csv',     'data/raw_dinuka_blue.csv',     'data/raw_nadun_blue.csv',      'data/raw_ravindu_blue.csv']
green_csv = ['data/raw_hansika_green.csv',  'data/raw_heshan_green.csv',    'data/raw_dinuka_green.csv',    'data/raw_nadun_green.csv',     'data/raw_ravindu_green.csv']
red_csv =   ['data/raw_hansika_red.csv',    'data/raw_heshan_red.csv',      'data/raw_dinuka_red.csv',      'data/raw_nadun_red.csv',       'data/raw_ravindu_red.csv']
    
train_data = np.concatenate((   preprocess(1, red_csv[0]),    preprocess(1, red_csv[1]),    preprocess(1, red_csv[2]),    
                                preprocess(2, green_csv[0]),  preprocess(2, green_csv[1]),  preprocess(2, green_csv[2]),  
                                preprocess(3, blue_csv[0]),   preprocess(3, blue_csv[1]),   preprocess(3, blue_csv[2])   
                            ), axis=0)

test_data = np.concatenate((    preprocess(1, red_csv[3]),    preprocess(1, red_csv[4]), 
                                preprocess(2, green_csv[3]),  preprocess(2, green_csv[4]),
                                preprocess(3, blue_csv[3]),   preprocess(3, blue_csv[4])
                           ), axis=0)



print train_data.shape
print test_data.shape

np.savetxt("train_data.csv", train_data, delimiter=",")
np.savetxt("test_data.csv", test_data, delimiter=",")