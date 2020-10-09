'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Main Function
- Import Dataset
- Generate Synthetic Dataset
- Evaluate the performances in three ways
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score

Inputs
- Dataset
- Network Parameters

Outputs
- time-series synthetic data
- Performances
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score
'''

#%% Necessary Packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
from matplotlib import pyplot

#%% Functions
# 1. Models
from tgan import tgan

# 2. Data Loading
from data_loading import google_data_loading, sine_data_generation, data_to_excel,load_real_samples, reorder_data, denormalize, unshuffle_data

# 3. Metrics
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

#%% Main Parameters
# Data
data_set = ['google','sine','TUD']
data_name = data_set[2]

# Experiments iterations
Iteration = 2
Sub_Iteration = 10
speed = 200         # 100,200,300,400,500
feed = 6           # 6,12

        
#%% Data Loading
seq_length = 24

if data_name == 'google':
    dataX = google_data_loading(seq_length)
elif data_name == 'sine':
    No = 10000
    F_No = 5
    dataX = sine_data_generation(No, seq_length, F_No)
elif data_name == 'TUD':
    dataX,dataXs, min, max, idx, data_true = load_real_samples(seq_length, speed, feed, False)
print(data_name + ' dataset is ready.')

#%% Newtork Parameters
parameters = dict()

parameters['hidden_dim'] = len(dataX[0][0,:]) * 4
parameters['num_layers'] = 3
parameters['iterations'] = 2
parameters['batch_size'] = 128
parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:]) 

#%% Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()

# Each Iteration
for it in range(Iteration):

    # Synthetic Data Generation
        dataX_hat = tgan(dataX,dataXs, parameters)   
          
        print('Finish Synthetic Data Generation')
    
        #%% Performance Metrics
        
        # 1. Discriminative Score
        Acc = list()
        for tt in range(Sub_Iteration):
            Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
            Acc.append(Temp_Disc)
        
        Discriminative_Score.append(np.mean(Acc))
        print('generated discriminative score')
        # 2. Predictive Performance
        MAE_All = list()
        for tt in range(Sub_Iteration):
            MAE_All.append(predictive_score_metrics (dataX, dataX_hat))
            
        Predictive_Score.append(np.mean(MAE_All))        
        print('generated predictive score')    
        
pyplot.scatter(np.linspace(1,24,24),dataX_hat[1][:,2], color='red')
pyplot.scatter(np.linspace(1,24,24),dataX[1][:,2], color='blue')

#%% 3. Visualization
PCA_Analysis (dataX, dataX_hat)
tSNE_Analysis (dataX, dataX_hat)

# Print Results
print('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
print('Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))

f = open('/home/awasthi/shrutarv/timegan/score.txt','w')
f.write('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)) + '\n' + 'Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))
f.close()

data_unshuffle = unshuffle_data(dataX_hat,idx)
data_reorder = reorder_data(data_unshuffle)
data_denormal = denormalize(data_reorder, min, max)
data_to_excel(data_denormal)
print('Data Saved')

fig, axs = pyplot.subplots(3,3)
fig.suptitle('Red- Generated data.    Blue- Real data.    x -axis is time')
#axs[0,0].scatter(np.linspace(1,50,50),data_denormal[:50,0], color = 'red', linewidths = 10)
#axs[0,0].scatter(np.linspace(1,50,50),data_true[:50,0], color = 'blue',linewidths = 10)
pyplot.subplot(331, xlabel = 'time',ylabel = 'time')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,0], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,0], color = 'blue')

pyplot.subplot(332, xlabel = 'time',ylabel = 'dx')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,1], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,1], color = 'blue')

#axs[0,1].scatter(np.linspace(1,50,50),data_denormal[:50,1], color = 'red', linewidth = 10)
#axs[0,1].scatter(np.linspace(1,50,50),data_true[:50,1], color = 'blue',linewidth = 10)
#axs[0,1].set_title('feed')

pyplot.subplot(333, xlabel = 'time',ylabel = 'dy')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,2], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,2], color = 'blue')

pyplot.subplot(333, xlabel = 'time',ylabel = 'fx')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,3], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,3], color = 'blue')

pyplot.subplot(334, xlabel = 'time',ylabel = 'fy')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,4], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,4], color = 'blue')

pyplot.subplot(335, xlabel = 'time',ylabel = 'fz')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,5], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,5], color = 'blue')

pyplot.subplot(335, xlabel = 'time',ylabel = 'wear')
pyplot.plot(np.linspace(1,50,50),data_denormal[:50,6], color = 'red')
pyplot.plot(np.linspace(1,50,50),data_true[:50,6], color = 'blue')
#axs[0,2].scatter(np.linspace(1,50,50),data_denormal[:50,2], color = 'red')
#axs[0,2].scatter(np.linspace(1,50,50),data_true[:50,2], color = 'blue')
#axs[0,2].set_title('time')
pyplot.savefig('S:/Job/test.png')
print('Plot saved')
# %%
