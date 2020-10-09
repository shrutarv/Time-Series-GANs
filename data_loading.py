'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
(1) Load Google dataset
- Transform the raw data to preprocessed data
(2) Generate Sine dataset

Inputs
(1) Google dataset
- Raw data
- seq_length: Sequence Length
(2) Sine dataset
- No: Sample Number
- T_No: Sequence Length
- F_No: Feature Number

Outputs
- time-series preprocessed data
'''

#%% Necessary Packages
import numpy as np
import pandas as pd

#%% Min Max Normalizer

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-12)

#%% Load Google Data
    
def google_data_loading (seq_length):

    # Load Google Data
    x = np.loadtxt('S:/Job/Time Series analysis/timegan/data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX
  
#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return dataX
    
def unshuffle_data(data,idx):
    data_unshuffle= list()
    for i in range(len(idx)):
        min_idx = np.argmin(idx)
        data_unshuffle.append(data[min_idx])
        #idx = np.delete(idx, (min_idx), axis =0)
        idx[min_idx] = 99999999
    return data_unshuffle    
    
def reorder_data(data):
    data = np.array(data)
    s = np.size(data,2)
    data_extract =   np.empty((1,s), int)
    for j in range(len(data[:])):
        if (j==0):
              
            for i in range (len(data[1,:,1])):
                data_extract = np.concatenate([data_extract,(data[0,i,:]).reshape(1,s)],axis=0)
        else:
            data_extract = np.concatenate([data_extract,(data[j,23,:]).reshape(1,s)],axis=0)
   # data_extract = np.concatenate([data_extract,(data[j,23,:]).reshape(1,9)],axis=0)
    data_extract = np.delete(data_extract, (0), axis=0)        
    return data_extract

def denormalize(data, min, max):
    for i in range(len(min)):
        data[:,i] = data[:,i]*(max[i]-min[i]) + min[i]
    return data

def data_to_excel(output, name):    
    #output = np.reshape(output,(len(output)*seq_length,9))
    p = "S:/Job/Time Series analysis/"
    ext = ".xlsx"
    path = p + name + ext
    writer = pd.ExcelWriter(path)
    df = pd.DataFrame(output)
    df.to_excel(writer, sheet_name='dataX')
    writer.save()
    
def load_real_samples(seq_length, speed, feed, unique_series = False):
	# load dataset, Pick specific columns, normalize, add a column to give
    # dummy variable to all the 10 conditions
	# convert from ints to floats
	#X = X.astype('float32')
	data_sorted = pd.read_csv("S:/Job/real-data-sample.csv")
	data_trans = data_sorted[['time_real','dx', 'dy','fx','fy','fz','wear_target_real']]
	data_static = data_sorted[['speed','feed']]
	data_trans = data_trans.values
	data_static = data_static.values
    
	#column = np.zeros((len(data_sorted),1))
	#data_sorted = np.append(data_sorted, column, axis=1)
	#labels = []
	#for i in range(len(data_sorted)):
	#	if(data_sorted[i,0]==100 and data_sorted[i,1]==6):
	#		data_sorted[i,9] = 1
	#	elif(data_sorted[i,0]==200 and data_sorted[i,1]==6):
	#		data_sorted[i,9] = 2
	#	elif(data_sorted[i,0]==300 and data_sorted[i,1]==6):
	#		data_sorted[i,9] = 3
	#	elif(data_sorted[i,0]==400 and data_sorted[i,1]==6):
	#		data_sorted[i,9] = 4            
	#	elif(data_sorted[i,0]==500 and data_sorted[i,1]==6):
	#	elif(data_sorted[i,0]==100 and data_sorted[i,1]==12):
		#	data_sorted[i,9] = 6
	#	elif(data_sorted[i,0]==200 and data_sorted[i,1]==12):
	#		data_sorted[i,9] = 7
	#	elif(data_sorted[i,0]==300 and data_sorted[i,1]==12):
	#		data_sorted[i,9] = 8
	#	elif(data_sorted[i,0]==400 and data_sorted[i,1]==12):
	#		data_sorted[i,9] = 9
	#	elif(data_sorted[i,0]==500 and data_sorted[i,1]==12):
	#		data_sorted[i,9] = 10
    
	if (unique_series==True):
		data_sorted = load_1_time_series(speed, feed, data_sorted) 
	data_trans[np.isnan(data_trans)]=0
	#labels = data_sorted[:,9]
	data_true = data_trans
	#data_sorted = data_sorted[::-1]
	min_t = np.min(data_trans, 0)
	max_t = np.max(data_trans, 0)
	data_trans = MinMaxScaler(data_trans)
	data_static = MinMaxScaler(data_static)

	# Build dataset
	dataX = []
	label = []
	dataXs = []
	# Cut data by sequence length
	for i in range(0, len(data_trans) - seq_length+1):
		_x = data_trans[i:i + seq_length]
		_xs = data_static[i:i + seq_length]
		#_l = labels[i:i+seq_length]
		dataX.append(_x)
		dataXs.append(_xs)
		#label.append(np.reshape(_l,(seq_length,1)))
		
	# Mix Data (to make it similar to i.i.d)
	idx = np.random.permutation(len(dataX))

	outputX = []
	#outputLabel=[]
	output_static = []
	for i in range(len(dataX)):
		outputX.append(dataX[idx[i]])
		#outputLabel.append(label[idx[i]])
		output_static.append(dataXs[idx[i]])
	return outputX,output_static,min_t,max_t,idx, data_true


    
def load_1_time_series(speed,feed,data):
    index = np.where((data[:,0]==speed) &  (data[:,1]==feed))
    output = data[index,:]
    output = output.reshape(len(output[0,:,0]),len(output[0,0,:])) 
    return output

def normalize(data):
    for i in range(2,9):
        minimum = np.min(data[:,i])
        if minimum<0:
            data[:,i] += np.abs(minimum)
        maximum = np.max(data[:,i])
        (data[:,i] - minimum)/(maximum - minimum)
    return data
