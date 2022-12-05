# In-vivo-dopamine-dynamics
Data and code for the manuscript: Behavioral encoding across timescales by region-specific dopamine dynamics.

*data/traces* cotains the full-length raw 415 nm and 470 nm traces for vehicle and cocaine-injected mice.

*data/spectral energy data* cotains data on spectral energy distribution as plotted on Figure 1C and D.

*data/behavioural tasks* cotains all non-CNO data on behavioural experiments as on Figure 2 and associated supporting figures.

The Jupyter notebook *spyder_notebook_behavioural tasks.py* (python v3.7.11) loads the data from both hemispheres, converts to dF/F0, fits a linear fit to correct for photobleaching and converts to zF. It also plots the spectal energy densities of the two regions and representative traces of the different bands. See manuscript for methodological details.

The Jupyter notebook *spyder_notebook_processing+energy densities.py* contains code to plot all non-CNO figures related to behaviour.

