import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
from scipy import stats
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pywt
from sklearn.linear_model import LinearRegression

DS_col = 'darkblue'
VS_col = 'darkred'
cmap = sns.color_palette([(0,0,139/255),(139/255,0,0)])

def ci_95_lin(x,y):
    # https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/
    # fit a curve to the data using a least squares 1st order polynomial fit
    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    fit = p(x)

    # get the coordinates for the fit curve
    c_y = [np.min(fit),np.max(fit)]
    c_x = [np.min(x),np.max(x)]

    # predict y values of origional data using the fit
    p_y = z[0] * x + z[1]

    # calculate the y-error (residuals)
    y_err = y -p_y

    # create series of new test x-values to predict for
    p_x = np.arange(np.min(x),np.max(x)+0.001,0.001)

    # now calculate confidence intervals for new test x-series
    mean_x = np.mean(x)         # mean of x
    n = len(x)              # number of samples in origional fit
    t = 1.943                # appropriate t value (where n=9, two tailed 95%)
    s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

    confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
                ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))

    # now predict y based on test x-values
    p_y = z[0]*p_x+z[1]

    # get lower and upper confidence limits based on predicted y and confidence intervals
    lower = p_y - abs(confs) 
    upper = p_y + abs(confs)
    return lower, upper, p_x, c_x, c_y


#%% Load data

NE_injVTA_DS = np.genfromtxt("data/behavioural tasks/Novel_injVTA_DS.csv",delimiter=",")
NE_injVTA_VS = np.genfromtxt("data/behavioural tasks/Novel_injVTA_VS.csv",delimiter=",")
NE_injVTA_Mov = np.genfromtxt("data/behavioural tasks/Novel_injVTA_Mov.csv",delimiter=",")
NE_injVTA_Dist = np.genfromtxt("data/behavioural tasks/Novel_injVTA_Dist.csv",delimiter=",")


SS_injVTA_DS = np.genfromtxt("data/behavioural tasks/SS_injVTA_DS.csv",delimiter=",")
SS_injVTA_VS = np.genfromtxt("data/behavioural tasks/SS_injVTA_VS.csv",delimiter=",")
SS_injVTA_Dist = np.genfromtxt("data/behavioural tasks/SS_injVTA_Dist.csv",delimiter=",")
SS_injVTA_Mov = np.genfromtxt("data/behavioural tasks/SS_injVTA_Mov.csv",delimiter=",")


# Novel Environment prep
NE_VS = NE_injVTA_VS[1:,:]
NE_DS = NE_injVTA_DS[1:,:]
NE_Mov = NE_injVTA_Mov[1:,:]
NE_Dist = NE_injVTA_Dist[1:,:]/10
                    
for i in range(9):
    NE_Mov[i,:] = np.convolve(NE_Mov[i,:], np.ones(100), 'full')[:1201]/100
    NE_Dist[i,:] = np.convolve(NE_Dist[i,:], np.ones(20), 'full')[:1201]/20


NE_DS_mean = np.mean(NE_DS,axis=0)
NE_DS_SEM = stats.sem(NE_DS,axis=0)
NE_VS_mean = np.mean(NE_VS,axis=0)
NE_VS_SEM = stats.sem(NE_VS,axis=0)
NE_Mov_mean = np.mean(NE_Mov,axis=0)
NE_Mov_SEM = stats.sem(NE_Mov,axis=0)
NE_Dist_mean = np.mean(NE_Dist,axis=0)
NE_Dist_SEM = stats.sem(NE_Dist,axis=0)


# Female bedding prep
SS_VS = SS_injVTA_VS[1:,:]
SS_DS = SS_injVTA_DS[1:,:]
SS_Dist = SS_injVTA_Dist[[1,3,4,5,6,7,8,9],:]/10 # Lost tracking for one mouse
SS_Mov = SS_injVTA_Mov[[1,3,4,5,6,7,8,9],:]


SS_DS_mean = np.mean(SS_DS,axis=0)
SS_DS_SEM = stats.sem(SS_DS,axis=0)
SS_VS_mean = np.mean(SS_VS,axis=0)
SS_VS_SEM = stats.sem(SS_VS,axis=0)

for i in range(8):
    SS_Mov[i,:] = np.convolve(SS_Mov[i,:], np.ones(100), 'full')[:1201]/100
    SS_Dist[i,:] = np.convolve(SS_Dist[i,:], np.ones(20), 'full')[:1201]/20

SS_Dist_mean = np.mean(SS_Dist,axis=0)
SS_Dist_SEM = stats.sem(SS_Dist,axis=0)
SS_Mov_mean = np.mean(SS_Mov,axis=0)
SS_Mov_SEM = stats.sem(SS_Mov,axis=0)

#%% Figure S4A
from matplotlib import lines


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(4, 3), dpi=400)
fig.text(0.5, 0.96, 'Novel environment', ha='center', fontsize = 10, rotation = 0)

for i in range(9):
    ax1.plot(NE_injVTA_DS[0,:],NE_DS[i,:]+i*6, color = "darkblue", lw = 0.8, zorder = 10)
    ax1.set_xlim(-10, 30) 
    ax1.set_xticks([-10,0,10,20,30])
    ax1.set_ylim(-6, 56)
    ax1.vlines(0,-10,100,ls = '-', color = [157/255, 176/255, 152/255], lw = 5)
    ax1.spines['top'].set_visible (False)
    ax1.spines['right'].set_visible (False)
    ax1.spines['left'].set_visible (False)
    ax1.set_yticks([])
    ax1.set_title("DS", fontsize = 10)
    ax1.set_xlabel("Time from entry (sec)")
    

    line = lines.Line2D([35.5,35.5], [22,27], lw=0.8, color='black', alpha=0.8)
    line.set_clip_on(False)
    ax1.add_line(line)
    
    ax2.plot(NE_injVTA_DS[0,:],NE_VS[i,:]+i*6, color = "darkred", lw = 0.8, zorder = 10)
    ax2.set_xlim(-10, 30) 
    ax2.set_xticks([-10,0,10,20,30])
    ax2.set_ylim(-6, 56) 
    ax2.vlines(0,-10,100,ls = '-', color = [157/255, 176/255, 152/255], lw = 5)
    ax2.spines['top'].set_visible (False)
    ax2.spines['right'].set_visible (False)
    ax2.spines['left'].set_visible (False)
    ax2.set_yticks([])
    ax2.set_title("VS", fontsize = 10)
    ax2.set_xlabel("Time from entry (sec)")
    
ax1.text(31.5,24.5,"5 zF", fontsize = 10, va = "center", color = "black", alpha=1, rotation = 90)  

plt.tight_layout(w_pad = 0)
#%% Figure S4B
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(4, 3), dpi=400)
fig.text(0.5, 0.96, 'Female bedding', ha='center', fontsize = 10, rotation = 0)
for i in range(9):
    ax1.plot(SS_injVTA_DS[0,:],SS_DS[i,:]+i*6, color = "darkblue", lw = 0.8, zorder = 10)
    ax1.set_xlim(-10, 30) 
    ax1.set_xticks([-10,0,10,20,30])
    ax1.set_ylim(-6, 56)
    ax1.vlines(0,-10,100,ls = '-', color = "peru", lw = 5)
    ax1.spines['top'].set_visible (False)
    ax1.spines['right'].set_visible (False)
    ax1.spines['left'].set_visible (False)
    ax1.set_yticks([])
    ax1.set_title("DS", fontsize = 10)
    ax1.set_xlabel("Time from 1st enc. (sec)")
    
    line = lines.Line2D([35.5,35.5], [22,27], lw=0.8, color='black', alpha=0.8)
    line.set_clip_on(False)
    ax1.add_line(line)
    
    ax2.plot(SS_injVTA_DS[0,:],SS_VS[i,:]+i*6, color = "darkred", lw = 0.8, zorder = 10)
    ax2.set_xlim(-10, 30) 
    ax2.set_xticks([-10,0,10,20,30])
    ax2.set_ylim(-4, 58) 
    ax2.vlines(0,-10,100,ls = '-', color = "peru", lw = 5)
    ax2.spines['top'].set_visible (False)
    ax2.spines['right'].set_visible (False)
    ax2.spines['left'].set_visible (False)
    ax2.set_yticks([])
    ax2.set_title("VS", fontsize = 10)
    ax2.set_xlabel("Time from 1st enc. (sec)")

ax1.text(31.5,24.5,"5 zF", fontsize = 10, va = "center", color = "black", alpha=1, rotation = 90)
    
plt.tight_layout(w_pad = 0)

#%% Figure 2A

x_min = -10
x_max = 30

y_min = -2
y_max = 5

fig, ax1 = plt.subplots(figsize=(2.5, 3), dpi=400) # subplots rather than plt will be used to fine-tune the output
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax1.set_title("Novel environment", fontsize = 10)
ax1.plot(100,100, lw = 1.5, color = DS_col, label = 'DS') 
ax1.plot(100,100, lw = 1.5, color = VS_col, label = 'VS')
ax1.plot(NE_injVTA_DS[0,:], NE_DS_mean, lw = 1.5, color = DS_col, label = 'DS') 
ax1.plot(NE_injVTA_DS[0,:], NE_VS_mean, lw = 1.5, color = VS_col, label = 'VS')
ax1.fill_between(NE_injVTA_DS[0,:], NE_DS_mean-NE_DS_SEM, NE_DS_mean+NE_DS_SEM, color = DS_col, alpha=0.4,lw=0.1)
ax1.fill_between(NE_injVTA_DS[0,:], NE_VS_mean-NE_VS_SEM, NE_VS_mean+NE_VS_SEM, color = VS_col, alpha=0.4,lw=0.1)
ax1.set_xlim(x_min,x_max) 
ax1.set_ylim(y_min,y_max) 
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax1.legend(("DS","VS"),loc = 'lower right', frameon = False, prop={'size': 9}, handlelength = 1, ncol = 2,
           bbox_to_anchor=(1.05,-0.02)) 
ax1.spines['right'].set_visible (False)

ax1.vlines(0,y_min,5,ls = '-', color = [157/255, 176/255, 152/255], lw = 8, zorder = 0)
ax1.set_ylabel("Mean zF", fontsize = 10, labelpad=-1)
# ax2.set_ylabel("cm/sec", fontsize = 10, labelpad=3)
ax1.set_xlabel("Time from entry (sec)", fontsize = 10)
# plt.title("Novel environment", fontsize = 10)
ax1.tick_params(axis='both', labelsize=10)
# ax2.tick_params(axis='both', labelsize=10)

fig.text(0.33, 0.77, 'Entry', ha='center', fontsize = 10, rotation = 90)
plt.tight_layout()

#%% Figure 2B
# Move and dist

fig, ax1 = plt.subplots(figsize=(2.8, 3), dpi=400) # subplots rather than plt will be used to fine-tune the output
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax1.set_title("Novel environment", fontsize = 10)
ax1.plot(NE_injVTA_DS[0,:], NE_Mov_mean, lw = 1.5, color = 'darkcyan', label = 'Mov.', ls = '-')
ax1.fill_between(NE_injVTA_DS[0,:], (NE_Mov_mean-NE_Mov_SEM), (NE_Mov_mean+NE_Mov_SEM), color = 'darkcyan', alpha=0.4,lw=0.1)
ax1.set_ylim(0,5)
ax1.set_xticks([-10,0,10,20,30])
# ax1.legend(("Speed",),loc = 'upper left', frameon = False, prop={'size': 10}, handlelength = 1) 
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax1.set_xlim(x_min,x_max) 
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylabel("Speed (cm/sec)", fontsize = 10, labelpad=3)
ax1.set_xlabel("Time from entry (sec)", fontsize = 10)
ax1.yaxis.label.set_color('darkcyan')
ax1.set_yticklabels(labels = [0,1,2,3,4,5], color = 'darkcyan')



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_zorder(1)
ax2.plot(NE_injVTA_DS[0,:], NE_Dist_mean, lw = 1.5, color = 'darkslateblue', label = 'Mov.', ls = '-')
ax2.fill_between(NE_injVTA_DS[0,:], (NE_Dist_mean-NE_Dist_SEM),
                 (NE_Dist_mean+NE_Dist_SEM), color = 'darkslateblue', alpha=0.4,lw=0.1)
ax2.set_ylim(0,25)
ax2.set_xlim(x_min,x_max)
ax2.set_xticks([-10,0,10,20,30])
# ax2.legend(("Distance",),loc = 'upper right', frameon = False, prop={'size': 10}, handlelength = 1.3) 
ax2.spines['top'].set_visible (False)
ax2.set_ylabel("Distance to mid. of nov. env. (cm)")
ax2.vlines(0,0,25,ls = '-', color = [157/255, 176/255, 152/255], lw = 8, zorder = 0)
ax2.yaxis.label.set_color('darkslateblue')
ax2.set_yticklabels(labels = [0,5,10,15,20,25,30,35], color = 'darkslateblue')

# fig.text(0.26, 0.775, 'Entry', ha='center', fontsize = 10, rotation = 90)
plt.tight_layout()

#%% Figure 4D
fig, ax1 = plt.subplots(figsize=(2.8, 3), dpi=400)
ax1.set_title("Novel environment\n(First 5 seconds)", fontsize = 10)
start = 600
stop = 1200

ax1.scatter(np.mean(NE_DS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1), color = "darkblue")
ax1.scatter(np.mean(NE_VS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1), color = "darkred")
ax1.legend(("DS","VS"), frameon = False, loc = "lower right")

lower, upper, p_x, c_x, c_y = ci_95_lin(np.mean(NE_DS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1))
ax1.plot(c_x, c_y, color = "darkblue")
ax1.fill_between(p_x, lower, upper, lw = 0, color = "darkblue",alpha = 0.1, zorder = 0)
print(stats.linregress(np.mean(NE_DS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1)))


lower, upper, p_x, c_x, c_y = ci_95_lin(np.mean(NE_VS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1))
ax1.plot(c_x, c_y, color = "darkred")
ax1.fill_between(p_x, lower, upper, lw = 0, color = "darkred",alpha = 0.1, zorder = 0)
print(stats.linregress(np.mean(NE_VS[:,start:stop],axis=1), np.mean(NE_Mov[:,start:stop],axis=1)))


ax1.set_xlabel("zF")
ax1.set_xlim(-0.1,1)
ax1.set_ylabel("cm/sec")
ax1.set_ylim(1,5)


ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Figure 2C

x_min = -10
x_max = 30

y_min = -2
y_max = 5

fig, ax1 = plt.subplots(figsize=(2.5, 3), dpi=400) # subplots rather than plt will be used to fiSS-tuSS the output
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax1.set_title("Female bedding", fontsize = 10)
ax1.plot(SS_injVTA_DS[0,:], SS_DS_mean, lw = 1.5, color = DS_col, label = 'DS') 
ax1.plot(SS_injVTA_DS[0,:], SS_VS_mean, lw = 1.5, color = VS_col, label = 'VS')
ax1.fill_between(SS_injVTA_DS[0,:], SS_DS_mean-SS_DS_SEM, SS_DS_mean+SS_DS_SEM, color = DS_col, alpha=0.4,lw=0.1)
ax1.fill_between(SS_injVTA_DS[0,:], SS_VS_mean-SS_VS_SEM, SS_VS_mean+SS_VS_SEM, color = VS_col, alpha=0.4,lw=0.1)
ax1.set_xlim(x_min,x_max) 
ax1.set_ylim(y_min,y_max) 
ax1.legend(("DS","VS"),loc = 'lower right', frameon = False, prop={'size': 9}, handlelength = 1, ncol = 2,
           bbox_to_anchor=(1.05,-0.02)) 
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)

ax1.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 8, zorder = 0)
ax1.set_ylabel("Mean zF", fontsize = 10, labelpad=-1)
ax1.set_xlabel("Time from encounter (sec)", fontsize = 10)
ax1.tick_params(axis='both', labelsize=10)
fig.text(0.33, 0.66, 'Encounter', ha='center', fontsize = 10, rotation = 90)
plt.tight_layout()

#%% Figure 2D
# Move and dist plot

fig, ax1 = plt.subplots(figsize=(2.8, 3), dpi=400) # subplots rather than plt will be used to fine-tune the output
ax1.set_zorder(2)
ax1.patch.set_visible(False)
ax1.set_title("Female bedding", fontsize = 10)
ax1.plot(SS_injVTA_DS[0,:]-0.5, SS_Mov_mean, lw = 1.5, color = 'darkcyan', label = 'Mov.', ls = '-')
ax1.fill_between(SS_injVTA_DS[0,:]-0.5, (SS_Mov_mean-SS_Mov_SEM), (SS_Mov_mean+SS_Mov_SEM), color = 'darkcyan', alpha=0.4,lw=0.1)
ax1.set_ylim(0,5)
ax1.set_xticks([-10,0,10,20,30])
# ax1.legend(("Speed",),loc = 'upper left', frameon = False, prop={'size': 10}, handlelength = 1) 
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax1.set_xlim(x_min,x_max) 
ax1.tick_params(axis='both', labelsize=10)
ax1.set_ylabel("Speed (cm/sec)", fontsize = 10, labelpad=3)
ax1.set_xlabel("Time from encounter (sec)", fontsize = 10)
# ax1.spines['right'].set_color('tab:cyan')
ax1.yaxis.label.set_color('darkcyan')
ax1.set_yticklabels(labels = [0,1,2,3,4,5], color = 'darkcyan')


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_zorder(1)
ax2.plot(SS_injVTA_DS[0,:]-0.5, SS_Dist_mean-5, lw = 1.5, color = 'darkslateblue', label = 'Mov.', ls = '-')
ax2.fill_between(SS_injVTA_DS[0,:]-0.5, (SS_Dist_mean-SS_Dist_SEM)-5,
                 (SS_Dist_mean+SS_Dist_SEM)-5, color = 'darkslateblue', alpha=0.4,lw=0.1)
ax2.set_ylim(0,35)
ax2.set_xlim(x_min,x_max)
ax2.set_xticks([-10,0,10,20,30])
# ax2.legend(("Distance",),loc = 'upper right', frameon = False, prop={'size': 10}, handlelength = 1.3) 
ax2.spines['top'].set_visible (False)
ax2.set_ylabel("Distance from bedding (cm)",labelpad=3, fontsize = 10)
ax2.vlines(0,0,35,ls = '-', color = "goldenrod", lw = 8, zorder = 0)
ax2.yaxis.label.set_color('darkslateblue')
ax2.set_yticklabels(labels = [0,5,10,15,20,25,30,35], color = 'darkslateblue')
plt.tight_layout()
# fig.text(0.27, 0.65, 'Encounter', ha='center', fontsize = 10, rotation = 90)
#%% Figure S4C
fig, ax1 = plt.subplots(figsize=(2, 3), dpi=400)
ax1.set_title("Novel environment\n(pre/post)", fontsize = 10)
ax1.set_ylabel("5 sec. average (zF)")

sns.swarmplot(data = [np.mean(NE_DS[:,500:600], axis = 1),np.mean(NE_DS[:,600:700], axis = 1),
                      np.mean(NE_VS[:,500:600], axis = 1),np.mean(NE_VS[:,600:700], axis = 1)],
              palette = ["darkblue", "darkblue", "darkred", "darkred"], s = 5, ax = ax1)

ax1.set_ylim(-2,4)
ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels(["DS, pre", "DS, post", "VS, pre", "VS, post"], rotation = 90)

for i in range(NE_DS.shape[0]):
    ax1.plot([0,1], [np.mean(NE_DS[i,500:600]), np.mean(NE_DS[i,600:700])], color = "dimgrey", alpha = 0.5, lw = 1)
    ax1.plot([2,3], [np.mean(NE_VS[i,500:600]), np.mean(NE_VS[i,600:700])], color = "dimgrey", alpha = 0.5, lw = 1)

print("DS: " + str(stats.ttest_rel(np.mean(NE_DS[:,500:600], axis = 1),np.mean(NE_DS[:,600:700], axis = 1))))
ax1.plot([0.1,0.9],[3,3],lw = 1.5, color = 'dimgrey')
ax1.annotate("n.s.", (0.5,3), textcoords="offset points", xytext=(0,3), 
             ha='center', color = 'dimgrey', weight='bold', fontsize = 8)
print("VS: " + str(stats.ttest_rel(np.mean(NE_VS[:,500:600], axis = 1),np.mean(NE_VS[:,600:700], axis = 1))))
ax1.plot([2.1,2.9],[3,3],lw = 1.5, color = 'dimgrey')
ax1.annotate("***", (2.5,3), textcoords="offset points", xytext=(0,0),
             ha='center', color = 'dimgrey', weight='bold')

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()

#%% Figure S4E
fig, ax1 = plt.subplots(figsize=(2, 3), dpi=400)
ax1.set_title("Female bedding\n(pre/post)", fontsize = 10)
ax1.set_ylabel("5 sec. average (zF)")

sns.swarmplot(data = [np.mean(SS_DS[:,400:500], axis = 1),np.mean(SS_DS[:,600:700], axis = 1),
                      np.mean(SS_VS[:,400:500], axis = 1),np.mean(SS_VS[:,600:700], axis = 1)],
              palette = ["darkblue", "darkblue", "darkred", "darkred"], s = 5, ax = ax1, clip_on = False)

ax1.set_ylim(-4,6)
ax1.set_xticks([0,1,2,3])
ax1.set_xticklabels(["DS, pre", "DS, post", "VS, pre", "VS, post"], rotation = 90)

for i in range(NE_DS.shape[0]):
    ax1.plot([0,1], [np.mean(SS_DS[i,400:500]), np.mean(SS_DS[i,600:700])], color = "dimgrey", alpha = 0.5, lw = 1)
    ax1.plot([2,3], [np.mean(SS_VS[i,400:500]), np.mean(SS_VS[i,600:700])], color = "dimgrey", alpha = 0.5, lw = 1)

print("DS: " + str(stats.ttest_rel(np.mean(SS_DS[:,400:500], axis = 1),np.mean(SS_DS[:,600:700], axis = 1))))
ax1.plot([0.1,0.9],[4.5,4.5],lw = 1.5, color = 'dimgrey')
ax1.annotate("n.s.", (0.5,4.5), textcoords="offset points", xytext=(0,3), 
             ha='center', color = 'dimgrey', weight='bold', fontsize = 8)
print("VS: " + str(stats.ttest_rel(np.mean(SS_VS[:,400:500], axis = 1),np.mean(SS_VS[:,600:700], axis = 1))))
ax1.plot([2.1,2.9],[4.5,4.5],lw = 1.5, color = 'dimgrey')
ax1.annotate("**", (2.5,4.5), textcoords="offset points", xytext=(0,0),
             ha='center', color = 'dimgrey', weight='bold')

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()
#%% Figure 2E
# VS
max_level = pywt.dwt_max_level(1201,"sym4")
NE_VS_split = np.array(pywt.mra(SS_VS[5,:], 'sym4', level = max_level,  transform='dwt'))

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=(2.5, 3), dpi=400, gridspec_kw={'height_ratios':[1,1,1,1,1]}) # subplots rather than plt will be used to fine-tune the output
ax1.set_title("Female bedding\nRepresentative VS", fontsize = 10)
ax1.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[:-1,400:], axis = 0), color = "darkred", lw = 1)
ax1.set_ylim(-2,5)
ax1.set_yticks([-2,5])
# ax1.set_ylabel("zF", labelpad = 0)
ax1.set_xlim(-10,30)
ax1.set_xticks([-10,0,10,20,30])
ax1.set_xticklabels([])
ax1.text(30,3.5,"Full", ha = "right", fontsize = 8)
ax1.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax2.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-3:-1,400:], axis = 0), color = "k", lw = 1.2)
ax2.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-3:-1,400:], axis = 0), color = [127/255,204/255,251/255], lw = 1)
ax2.set_ylim(-2,5)
ax2.set_yticks([-2,5])
# ax2.set_ylabel("zF", labelpad = 0)
ax2.set_xlim(-10,30)
ax2.set_xticks([-10,0,10,20,30])
ax2.set_xticklabels([])
ax2.text(30,3.5,"2.5-10 Hz", ha = "right", fontsize = 8)
ax2.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax2.spines['top'].set_visible (False)
ax2.spines['right'].set_visible (False)
ax3.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-5:-3,400:], axis = 0), color = "k", lw = 1.2)
ax3.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-5:-3,400:], axis = 0), color = [137/255,231/255,180/255], lw = 1)
ax3.set_ylim(-2,5)
ax3.set_yticks([-2,5])
ax3.set_ylabel("zF", labelpad = 2)
ax3.set_xlim(-10,30)
ax3.set_xticks([-10,0,10,20,30])
ax3.set_xticklabels([])
ax3.text(30,3.5,"0.6-2.5 Hz", ha = "right", fontsize = 8)
ax3.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax3.spines['top'].set_visible (False)
ax3.spines['right'].set_visible (False)
ax4.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-7:-5,400:], axis = 0), color = "k", lw = 1.2)
ax4.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-7:-5,400:], axis = 0), color = [238/255,251/255,141/255], lw = 1)
ax4.set_ylim(-2,5)
ax4.set_yticks([-2,5])
# ax4.set_ylabel("zF", labelpad = 0)
ax4.set_xlim(-10,30)
ax4.set_xticks([-10,0,10,20,30])
ax4.set_xticklabels([])
ax4.text(30,3.5,"0.15-0.6 Hz", ha = "right", fontsize = 8)
ax4.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax4.spines['top'].set_visible (False)
ax4.spines['right'].set_visible (False)
ax5.plot(np.linspace(-10,30,801),NE_VS_split[0,400:], color = "k", lw = 1.2)
ax5.plot(np.linspace(-10,30,801),NE_VS_split[0,400:], color = [211/255,199/255,180/255], lw = 1)
ax5.set_ylim(-2,5)
ax5.set_yticks([-2,5])
# ax5.set_ylabel("zF", labelpad = 0)
ax5.set_xlim(-10,30)
ax5.set_xticks([-10,0,10,20,30])
ax5.set_xlabel("Time from entry (sec)")
ax5.text(30,3.5,"<0.15 Hz", ha = "right", fontsize = 8)
ax5.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)

ax5.spines['top'].set_visible (False)
ax5.spines['right'].set_visible (False)

fig.tight_layout(h_pad = 0.1)
#%% Figure 2E
# DS
max_level = pywt.dwt_max_level(1201,"sym4")
NE_VS_split = np.array(pywt.mra(SS_DS[5,:], 'sym4', level = max_level,  transform='dwt'))

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=(2.5, 3), dpi=400, gridspec_kw={'height_ratios':[1,1,1,1,1]}) # subplots rather than plt will be used to fine-tune the output
ax1.set_title("Female bedding\nRepresentative DS", fontsize = 10)
ax1.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[:-1,400:], axis = 0), color = "darkblue", lw = 1)
ax1.set_ylim(-2,5)
ax1.set_yticks([-2,5])
# ax1.set_ylabel("zF", labelpad = 0)
ax1.set_xlim(-10,30)
ax1.set_xticks([-10,0,10,20,30])
ax1.set_xticklabels([])
ax1.text(30,3.5,"Full", ha = "right", fontsize = 8)
ax1.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax1.spines['top'].set_visible (False)
ax1.spines['right'].set_visible (False)
ax2.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-3:-1,400:], axis = 0), color = "k", lw = 1.2)
ax2.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-3:-1,400:], axis = 0), color = [127/255,204/255,251/255], lw = 1)
ax2.set_ylim(-2,5)
ax2.set_yticks([-2,5])
# ax2.set_ylabel("zF", labelpad = 0)
ax2.set_xlim(-10,30)
ax2.set_xticks([-10,0,10,20,30])
ax2.set_xticklabels([])
ax2.text(30,3.5,"2.5-10 Hz", ha = "right", fontsize = 8)
ax2.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax2.spines['top'].set_visible (False)
ax2.spines['right'].set_visible (False)
ax3.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-5:-3,400:], axis = 0), color = "k", lw = 1.2)
ax3.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-5:-3,400:], axis = 0), color = [137/255,231/255,180/255], lw = 1)
ax3.set_ylim(-2,5)
ax3.set_yticks([-2,5])
ax3.set_ylabel("zF", labelpad = 2)
ax3.set_xlim(-10,30)
ax3.set_xticks([-10,0,10,20,30])
ax3.set_xticklabels([])
ax3.text(30,3.5,"0.6-2.5 Hz", ha = "right", fontsize = 8)
ax3.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax3.spines['top'].set_visible (False)
ax3.spines['right'].set_visible (False)
ax4.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-7:-5,400:], axis = 0), color = "k", lw = 1.2)
ax4.plot(np.linspace(-10,30,801),np.sum(NE_VS_split[-7:-5,400:], axis = 0), color = [238/255,251/255,141/255], lw = 1)
ax4.set_ylim(-2,5)
ax4.set_yticks([-2,5])
# ax4.set_ylabel("zF", labelpad = 0)
ax4.set_xlim(-10,30)
ax4.set_xticks([-10,0,10,20,30])
ax4.set_xticklabels([])
ax4.text(30,3.5,"0.15-0.6 Hz", ha = "right", fontsize = 8)
ax4.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax4.spines['top'].set_visible (False)
ax4.spines['right'].set_visible (False)
ax5.plot(np.linspace(-10,30,801),NE_VS_split[0,400:], color = "k", lw = 1.2)
ax5.plot(np.linspace(-10,30,801),NE_VS_split[0,400:], color = [211/255,199/255,180/255], lw = 1)
ax5.set_ylim(-2,5)
ax5.set_yticks([-2,5])
# ax5.set_ylabel("zF", labelpad = 0)
ax5.set_xlim(-10,30)
ax5.set_xticks([-10,0,10,20,30])
ax5.set_xlabel("Time from entry (sec)")
ax5.text(30,3.5,"<0.15 Hz", ha = "right", fontsize = 8)
ax5.vlines(0,-2,5,ls = '-', color = "goldenrod", lw = 5, zorder = 0)
ax5.spines['top'].set_visible (False)
ax5.spines['right'].set_visible (False)

fig.tight_layout(h_pad = 0.1)
