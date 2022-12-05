# =============================================================================
# RAW DATA FOR PREPROCESSING
# =============================================================================
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import glob
DS_col = 'darkblue'
VS_col = 'darkred'

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

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (6,6), dpi = 400)
ax1.set_title("DS; cocaine", fontsize = 12)
ax2.set_title("VS; cocaine", fontsize = 12)


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
#%%
# =============================================================================
# ENERGY DENSITIES
# =============================================================================

Energy_combined_DS = np.genfromtxt("data/spectral energy data/Energy_DS.csv",delimiter=",")
Energy_combined_VS = np.genfromtxt("data/spectral energy data/Energy_VS.csv",delimiter=",")

DS_sum = Energy_combined_DS[:-1,1:].sum(axis=0)
Energy_combined_DS_relative = Energy_combined_DS[:-1,1:]/DS_sum[:,None].T
Energy_combined_DS_mean = np.mean(Energy_combined_DS_relative,axis=1)
Energy_combined_DS_SEM = stats.sem(Energy_combined_DS_relative,axis=1)
Energy_combined_DS_SEM = np.vstack([np.zeros(15),Energy_combined_DS_SEM])

VS_sum = Energy_combined_VS[:-1,1:].sum(axis=0)
Energy_combined_VS_relative = Energy_combined_VS[:-1,1:]/VS_sum[:,None].T
Energy_combined_VS_mean = np.mean(Energy_combined_VS_relative,axis=1)
Energy_combined_VS_SEM = stats.sem(Energy_combined_VS_relative,axis=1)
Energy_combined_VS_SEM = np.vstack([np.zeros(15),Energy_combined_VS_SEM])

fig, ax1 = plt.subplots(figsize=(3.6, 3), dpi=400)

ax1.bar(np.linspace(0,14,15)+0.21, Energy_combined_DS_mean*100, width = 0.42, yerr = Energy_combined_DS_SEM*100, color = DS_col)
ax1.bar(np.linspace(0,14,15)-0.21, Energy_combined_VS_mean*100, width = 0.42, yerr = Energy_combined_VS_SEM*100, color = VS_col)
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax1.set_xticks(np.linspace(0,15,16)-0.5)


xlabels = ["10","5.0","2.5","1.25","0.62","0.31",
            "0.15", "0.08","0.04", "0.02","0.01",
            "0.005","0.003","0.001", "0.0006","0.0003"]

ax1.set_xticklabels(xlabels,rotation=90, ha = "center", va = "top",fontsize = 10)
plt.xlabel("Frequency band (~Hz)", fontsize = 10)
plt.ylabel("Relative energy density (%)", fontsize = 10)
plt.title("Energy density at frequency bands", fontsize = 10)
plt.ylim(0,25)
plt.yticks([0,5,10,15,20,25], fontsize = 10)
plt.xlim(-1,15)
plt.legend(("DS","VS"),frameon = False, loc = "center left", prop={'size': 10}, handlelength =0.8)

ax1.invert_xaxis()

ax1.text(3.75, 24.3, 'Rapid domain', ha='center', va ="top", fontsize=10)
ax1.text(11.75, 24.3, 'Slow domain', ha='center', va ="top", fontsize=10)

x = np.linspace(-1,15,10)
y = np.linspace(0,25,10)
z = np.ones((10,10))
for i in range(10):
    z[:,i] = z[:,i]*i
num_bars = 100  # more bars = smoother gradient

plt.vlines(8.5,0,25, color = [0.3,0.3,0.3], ls = "--", lw = 1)
plt.contourf(x, y, z[:,::], num_bars, cmap ="terrain", alpha = 0.5, vmin = -0.7, vmax = 9)

plt.tight_layout()

#%% Traces split in energy bands
time_freq_dom_DS = np.genfromtxt("data/spectral energy data/time_freq_domain_DS.csv",delimiter=",")
time_freq_dom_VS = np.genfromtxt("data/spectral energy data/time_freq_domain_VS.csv",delimiter=",")

fig, axes = plt.subplots(11,1,figsize=(3, 6.05), dpi=400, gridspec_kw={'height_ratios':[1,1,1,1,1,0.1,1,1,1,1,1]}) # subplots rather than plt will be used to fine-tune the output

axes.reshape(-1)[0].set_title("DS", fontsize = 10)
axes.reshape(-1)[0].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[1:,:], axis = 0), color = "darkblue", lw = 1)
axes.reshape(-1)[0].text(835,3,"Full", ha = "right", fontsize = 8)

axes.reshape(-1)[1].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[2:4,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[1].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[2:4,:], axis = 0), color = [127/255,204/255,251/255], lw = 1)
axes.reshape(-1)[1].text(835,3,"2.5-10 Hz", ha = "right", fontsize = 8)

axes.reshape(-1)[2].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[4:6,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[2].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[4:6,:], axis = 0), color = [137/255,231/255,180/255], lw = 1)
axes.reshape(-1)[2].text(835,3,"0.6-2.5 Hz", ha = "right", fontsize = 8)
axes.reshape(-1)[2].set_ylabel("zF", labelpad = 0)

axes.reshape(-1)[3].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[6:8,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[3].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[6:8,:], axis = 0), color = [238/255,251/255,141/255], lw = 1)
axes.reshape(-1)[3].text(835,3,"0.15-0.6 Hz", ha = "right", fontsize = 8)

axes.reshape(-1)[4].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[8:,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[4].plot(time_freq_dom_DS[0,:], np.sum(time_freq_dom_DS[8:,:], axis = 0), color = [211/255,199/255,180/255], lw = 1)
axes.reshape(-1)[4].text(835,3,"<0.15 Hz", ha = "right", fontsize = 8)


axes.reshape(-1)[6].set_title("VS", fontsize = 10)
axes.reshape(-1)[6].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[1:,:], axis = 0), color = "darkred", lw = 1)
axes.reshape(-1)[6].text(835,3,"Full", ha = "right", fontsize = 8)

axes.reshape(-1)[7].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[2:4,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[7].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[2:4,:], axis = 0), color = [127/255,204/255,251/255], lw = 1)
axes.reshape(-1)[7].text(835,3,"2.5-10 Hz", ha = "right", fontsize = 8)

axes.reshape(-1)[8].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[4:6,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[8].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[4:6,:], axis = 0), color = [137/255,231/255,180/255], lw = 1)
axes.reshape(-1)[8].text(835,3,"0.6-2.5 Hz", ha = "right", fontsize = 8)
axes.reshape(-1)[8].set_ylabel("zF", labelpad = 0)

axes.reshape(-1)[9].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[6:8,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[9].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[6:8,:], axis = 0), color = [238/255,251/255,141/255], lw = 1)
axes.reshape(-1)[9].text(835,3,"0.15-0.6 Hz", ha = "right", fontsize = 8)

axes.reshape(-1)[10].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[8:,:], axis = 0), color = "k", lw = 1.2)
axes.reshape(-1)[10].plot(time_freq_dom_VS[0,:], np.sum(time_freq_dom_VS[8:,:], axis = 0), color = [211/255,199/255,180/255], lw = 1)
axes.reshape(-1)[10].text(835,3,"<0.15 Hz", ha = "right", fontsize = 8)

axes.reshape(-1)[5].axis('off')

for i in [0,1,2,3,4,6,7,8,9,10]:
    axes.reshape(-1)[i].set_ylim(-2,4)
    axes.reshape(-1)[i].set_yticks([-2,4])
    axes.reshape(-1)[i].set_xlim(815,815+20)
    axes.reshape(-1)[i].set_xticks([815,825,835])
    axes.reshape(-1)[i].set_xticklabels([])
    axes.reshape(-1)[i].spines["top"].set_visible(False)
    axes.reshape(-1)[i].spines["right"].set_visible(False)
# for i in np.arange(0,8,2):
#     axes.reshape(-1)[i].set_yticks([-2,4])
#     axes.reshape(-1)[i].set_yticklabels([-2,4])
    
    
# axes.reshape(-1)[6].set_xticklabels([0,10,20])
# axes.reshape(-1)[6].set_xlabel("Seconds", labelpad = 1)
axes.reshape(-1)[10].set_xticklabels([0,10,20])
axes.reshape(-1)[10].set_xlabel("Seconds", labelpad = 1)

plt.tight_layout(h_pad = -0.1)