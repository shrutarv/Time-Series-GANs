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
    return numerator / (denominator + 1e-7)

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
    

def data_to_excel(output, seq_length, name):    
    output = np.reshape(output,(len(output)*seq_length,9))
    p = "S:/Job/Time Series analysis/"
    ext = ".xlsx"
    path = p + name + ext
    writer = pd.ExcelWriter(path)
    df = pd.DataFrame(output)
    df.to_excel(writer, sheet_name='dataX')
    writer.save()
    
def load_real_samples(seq_length, speed, feed, unique_series = True):
	# load dataset, Pick specific columns, normalize, add a column to give
    # dummy variable to all the 10 conditions
	# convert from ints to floats
	#X = X.astype('float32')
	data_sorted = pd.read_csv("S:/Job/real-data-sample.csv")
	#data_sorted = data[['dx', 'dy','fx','fy','fz','wear_target_real']]
	data_sorted = data_sorted.values
	if (unique_series==True):
		data_sorted = load_1_time_series(speed, feed, data_sorted) 
	data_sorted[np.isnan(data_sorted)]=0
	data_sorted = data_sorted[::-1]
	data_sorted = MinMaxScaler(data_sorted)

	# Build dataset
	dataX = []
    
	# Cut data by sequence length
	for i in range(0, len(data_sorted) - seq_length):
		_x = data_sorted[i:i + seq_length]
		dataX.append(_x)
		
	# Mix Data (to make it similar to i.i.d)
	idx = np.random.permutation(len(dataX))

	outputX = []
	for i in range(len(dataX)):
		outputX.append(dataX[idx[i]])

	return outputX

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
