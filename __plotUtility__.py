import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import arviz as az
import corner


# Define custom color list with colorblind friendlyness for cornerplots
color_list = ['none', '#7acfff', '#00c48f', '#0093e6']

# Matplotlib preferences
plt.rc('font',      size=14)        # controls default text sizes
plt.rc('axes',      titlesize=16)   # fontsize of the axes title
plt.rc('axes',      labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick',     labelsize=16)   # fontsize of the tick labels
plt.rc('ytick',     labelsize=16)   # fontsize of the tick labels
plt.rc('legend',    fontsize=16)    # legend fontsize
plt.rc('figure',    titlesize=16)   # fontsize of the figure title
plt.rc('figure',    dpi=75)         # Changed to 100 for some version >3.5.2
plt.rc('axes',      grid=True)      # Default set grids to true




########################################
########################################
### P L O T T I N G    U T I L I T Y ###
########################################
########################################

# function for basic plotting using different colors and markers by source
def plot_data(ax, data, plotDict, axisDict, plotSettings, interpolateDict, legendCol=2, legendLoc='best'):
    # Extract relevant keys
    key_x    = plotDict['x']
    key_y    = plotDict['y']
    key_xerr = plotDict['x_err']
    key_yerr = plotDict['y_err']
    
    # Loop over entries
    for index in range(len(data)):
        # Makes sure not to get duplicate labels
        if data['source'][index] not in ax.get_legend_handles_labels()[1]:
            myLabel = data['source'][index]
        else:
            myLabel = ''


        # Interpolation of function (used for residuals)
        fitInterp = 0
        if interpolateDict['subtract']:
            fitInterp = np.interp(data[key_x][index], interpolateDict['x_linspace'], interpolateDict['function'])
            

        # Relevant data sets to plot
        xval = data[key_x][index]
        yval = data[key_y][index] - fitInterp
        xerr = data[key_xerr][index]
        yerr = data[key_yerr][index]

        # Plot data points
        if interpolateDict['ratio']:
            yval = 100 * (yval/fitInterp) # Change of difference to relative difference
            yerr = 100 * (yerr/fitInterp)

        ax.errorbar(xval, yval, xerr = xerr, yerr = yerr,
                    marker = plotSettings['markers'][data['source'][index]], color = plotSettings['colors'][data['source'][index]],
                    markersize = plotSettings['markersizes'][data['source'][index]], label = myLabel)

    # Axis and legend
    ax.legend(loc=legendLoc, ncol=legendCol)
    #ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax.set_xlabel(axisDict['label_x'])
    ax.set_ylabel(axisDict['label_y'])
    if axisDict['range_x'] != 0:
        ax.set_xlim(axisDict['range_x'][0], axisDict['range_x'][1])
    if axisDict['range_y'] != 0:
        ax.set_ylim(axisDict['range_y'][0], axisDict['range_y'][1])



# Plot the log of eff with potentially a fit
def plot_logEff(ax, data, fitDict, plotSettings, legendCol=3, legendLoc='best'):
    # Prepare dictionaries to pass to plotData
    plotDict = {
        'x' : 'E', 'x_err' : 'E_err',
        'y' : 'logeff', 'y_err' : 'logeff_err'
    }

    interpolateDict = {
        'subtract' : False, 'ratio': False, 'function' : 0, 'x_linspace' : 0
    }

    min_y = min(data[plotDict['y']])
    max_y = max(data[plotDict['y']])
    
    axisDict = {
        'label_x' : "Energy (keV)", 'range_x' : 0,
        'label_y' : r'$log(\varepsilon)$', 'range_y' : [min_y - 0.2*abs(max_y-min_y), max_y + 0.2*abs(max_y-min_y)]
    }

    # If requested, also show fit and confidence interval
    if fitDict['show']:
        ax.plot(fitDict['xvals'], fitDict['yvals'], 'r-', label="fit")
        ax.fill_between(fitDict['xvals'], fitDict['lower'], fitDict['upper'], color="grey", alpha=0.2, label=r'$1\sigma$')

    plot_data(ax, data, plotDict, axisDict, plotSettings, interpolateDict, legendCol=legendCol, legendLoc=legendLoc)



# Plot the eff with potentially a fit
def plot_eff(ax, data, fitDict, plotSettings, legendCol=3, legendLoc='best', scaled=False):
    # Prepare dictionaries to pass to plotData
    plotDict = {
        'x' : 'E', 'x_err' : 'E_err',
        'y' : 'eff', 'y_err' : 'eff_err'
    }

    if scaled:
        plotDict['y']     = 'recalc_eff'
        plotDict['y_err'] = 'recalc_eff_err'
    
    axisDict = {
        'label_x' : "Energy (keV)", 'range_x' : 0,
        'label_y' : r'$\varepsilon$', 'range_y' : [0, 1.2*max(data['eff'])]
    }

    interpolateDict = {
        'subtract' : False, 'ratio': False, 'function' : 0, 'x_linspace' : 0
    }

    # If requested, also show fit and confidence interval
    if fitDict['show']:
        ax.plot(fitDict['xvals'], fitDict['yvals'], 'r-', label="fit")
        ax.fill_between(fitDict['xvals'], fitDict['lower'], fitDict['upper'], color="grey", alpha=0.2, label=r'$1\sigma$')

    plot_data(ax, data, plotDict, axisDict, plotSettings, interpolateDict, legendCol=legendCol, legendLoc=legendLoc)
    



# Plot the relative residuals compared to the fit function (value/fit - 1)
def plot_relResid_eff(ax, data, fitDict, plotSettings, legendCol=3, legendLoc='best', scaled=False):    
    # Prepare dictionaries to pass to plotData
    plotDict = {
    'x' : 'E', 'x_err' : 'E_err',
    'y' : 'eff', 'y_err' : 'eff_err'
    }

    if scaled:
        plotDict['y']     = 'recalc_eff'
        plotDict['y_err'] = 'recalc_eff_err'

        
    interpolateDict = { # Subtract fit to get residuals
        'subtract' : True, 'ratio': True, 'x_linspace' : fitDict['xvals'], 'function' : fitDict['yvals']
    }

    interp = np.interp(data[plotDict['x']], interpolateDict['x_linspace'], interpolateDict['function'])
    values = 100 * (data[plotDict['y']] - interp)/interp
    min_y  = min(values)
    max_y  = max(values)
    
    #min_y = min(data[plotDict['y']] - np.interp(data[plotDict['x']], interpolateDict['x_linspace'], interpolateDict['function']))
    #max_y = max(data[plotDict['y']] - np.interp(data[plotDict['x']], interpolateDict['x_linspace'], interpolateDict['function']))
    
    axisDict = {
        'label_x' : "Energy (keV)", 'range_x' : 0,
        'label_y' : 'Relative residual (%)', 'range_y' : [min_y - 1.0*abs(max_y-min_y), max_y + 1.0*abs(max_y-min_y)]
    }

    # If requested, draw confidence band
    if fitDict['show']:
        ax.fill_between(fitDict['xvals'], 100*(fitDict['lower'] - fitDict['yvals'])/fitDict['yvals'], 
                        100*(fitDict['upper'] - fitDict['yvals'])/fitDict['yvals'], color="grey", alpha=0.2, label=r'$1\sigma$')

    plot_data(ax, data, plotDict, axisDict, plotSettings, interpolateDict, legendCol=legendCol, legendLoc=legendLoc)
    ax.legend().set_visible(False)




# Make a nice cornerplot
def corner_plot(trace, var_names, N, figsize=(12, 12)):
    fig, axes = plt.subplots(N, N, figsize=figsize)
    
    fig = corner.corner(trace, fig=fig, var_names=var_names, 
                        plot_datapoints=True, plot_density=False, plot_contours=True,
                        levels=[0.6827, 0.9545, 0.9973], smooth=0.8,
                        quantiles=[0.159, 0.5, 0.841], bins=50, 
                        fill_contours=True, contourf_kwargs={'colors': color_list},
                        show_titles=True, title_kwargs={'fontsize': 12}, title_fmt='.1e')

    axes = fig.get_axes()

    for i, ax in enumerate(axes):
        row, col = divmod(i, N)  # Determine the row and column of the subplot

        # Apply labels and scientific notation only to the left column (y-axis)
        if col == 0:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.offsetText.set_position((-0.15, 0.9))  # Adjust as needed
            ax.yaxis.offsetText.set_rotation(0)
            ax.yaxis.set_label_coords(-0.5, 0.5)
        else:
            ax.set_yticklabels([])  # Remove y tick labels for other columns

        # Apply labels and scientific notation only to the bottom row (x-axis)
        if row == N - 1:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.xaxis.set_label_coords(0.5, -0.5)
            ax.xaxis.offsetText.set_position((1.1, 0.))  # Adjust as needed
            ax.xaxis.offsetText.set_rotation(90)
        else:
            ax.set_xticklabels([])



    
    #for ax in fig.get_axes():
    #    ax.xaxis.set_label_coords(0.5, -0.5)
    #    ax.yaxis.set_label_coords(-0.4, 0.5)        

    plt.gcf().subplots_adjust(left=0.175)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.subplots_adjust(wspace=0.225, hspace=0.225, right=0.875)

    
    #plt.tight_layout()


