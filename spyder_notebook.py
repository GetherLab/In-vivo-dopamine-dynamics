#%% Load data
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load file names
trace_files_DS = glob.glob("data/traces/DS/*")
trace_files_VS = glob.glob("data/traces/VS/*")

# Load time points and pharmacological type
# (See file for info)
time_points = np.genfromtxt("data/timepoints.csv", delimiter = ",", skip_header=1,  usecols = (1, 2, 3, 4), dtype = int)
pharmacology = np.genfromtxt("data/timepoints.csv", delimiter = ",", skip_header=1,  usecols = (5), dtype = str)


# Load photometry traces
# First column is 470 nm, second is 415 nm
raw_data_DS = []
for i in trace_files_DS:
    raw_data_DS.append(np.genfromtxt(i,delimiter=","))
    
raw_data_VS = []
for i in trace_files_VS:
    raw_data_VS.append(np.genfromtxt(i,delimiter=","))
#%% Preprocess data

# Convert to dF/F0 and collect both regions in to one variable
dF_F0 = []
for i in range(len(raw_data_DS)):
    dF_DS = (raw_data_DS[i][time_points[i,0]:time_points[i,3],0] -\
             raw_data_DS[i][time_points[i,0]:time_points[i,3],1])/\
             raw_data_DS[i][time_points[i,0]:time_points[i,3],1]
             
    dF_VS = (raw_data_VS[i][time_points[i,0]:time_points[i,3],0] -\
             raw_data_VS[i][time_points[i,0]:time_points[i,3],1])/\
             raw_data_VS[i][time_points[i,0]:time_points[i,3],1]
             
    dF_F0.append(np.vstack((dF_DS,dF_VS)))
    
# Correct for photobleach with a linear fit
zF = []
for i in range(len(raw_data_DS)):
    # DS
    pre_inj = np.vstack((np.arange(0,time_points[i,1]),
                         dF_F0[i][0,:time_points[i,1]]))
    
    end_of_trace = np.vstack((np.arange(dF_F0[i].shape[1]-12000,dF_F0[i].shape[1]),
                         dF_F0[i][0,dF_F0[i].shape[1]-12000:dF_F0[i].shape[1]]))
    
    combined_fit_data = np.hstack((pre_inj,end_of_trace))
    
    fit = np.polyfit(combined_fit_data[0,:], combined_fit_data[1,:],1)
    correction = np.arange(0,dF_F0[i].shape[1])*fit[0]+fit[1]
    
    corr_DS = dF_F0[i][0,:] - correction
    zF_DS = (corr_DS - np.mean(corr_DS[time_points[i,1]-7200:time_points[i,1]-1200]))\
        /np.std(corr_DS[time_points[i,1]-7200:time_points[i,1]-1200])
    
    # VS
    pre_inj = np.vstack((np.arange(0,time_points[i,1]),
                         dF_F0[i][1,:time_points[i,1]]))
    
    end_of_trace = np.vstack((np.arange(dF_F0[i].shape[1]-12000,dF_F0[i].shape[1]),
                         dF_F0[i][1,dF_F0[i].shape[1]-12000:dF_F0[i].shape[1]]))
    
    combined_fit_data = np.hstack((pre_inj,end_of_trace))
    
    fit = np.polyfit(combined_fit_data[0,:], combined_fit_data[1,:],1)
    correction = np.arange(0,dF_F0[i].shape[1])*fit[0]+fit[1]
    
    corr_VS = dF_F0[i][1,:] - correction
    zF_VS = (corr_VS - np.mean(corr_VS[time_points[i,1]-7200:time_points[i,1]-1200]))\
        /np.std(corr_VS[time_points[i,1]-7200:time_points[i,1]-1200])
        
    # Combine
    zF.append(np.vstack((zF_DS,zF_VS)))

#%% Plot the data
veh_mice = np.where(pharmacology == "vehicle")[0]

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (4,4), dpi = 400)
ax1.set_title("DS; cocaine", fontsize = 10)
ax2.set_title("VS; cocaine", fontsize = 10)


for count, idx in enumerate(veh_mice):
    if count == 7:
        ax1.plot(zF[idx][0,:] + count*20, color = "dimgrey", lw = 0.2, alpha = 0.5)
        ax1.set_ylim(-10,190)
        ax1.set_xlim(0,180000)
    
        ax2.plot(zF[idx][1,:] + count*20, color = "dimgrey", lw = 0.5, alpha = 0.5)
        ax2.set_ylim(-10,190)
        ax2.set_xlim(0,180000)
    else:
        ax1.plot(zF[idx][0,:] + count*20, color = "darkblue", lw = 0.2)
        ax1.set_ylim(-10,190)
        ax1.set_xlim(0,180000)
    
        ax2.plot(zF[idx][1,:] + count*20, color = "darkred", lw = 0.5)
        ax2.set_ylim(-10,190)
        ax2.set_xlim(0,180000)

# Add scale bars
ax1.plot([185000,185000], [175,185], color = "black", lw = 1.5, clip_on = False)
ax1.plot([185000,197000], [175,175], color = "black", lw = 1.5, clip_on = False)

# Remove axis
ax1.set_axis_off() 
ax2.set_axis_off() 
