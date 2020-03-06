import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # for "Total SM & uncertainty" merged legend handle
from matplotlib.lines import Line2D # for dashed line in legend
from matplotlib.ticker import AutoMinorLocator,LogLocator,LogFormatterSciNotation # for minor ticks  
import os.path

import labelfile
import HZZAnalysis
import HZZSamples
import HZZHistograms


class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)


def plot_data(data):

    signal_format = 'hist' # 'line' or 'hist' or None
    Total_SM_label = False # for Total SM black line in plot and legend
    plot_label = r'$H \rightarrow ZZ^* \rightarrow \ell\ell\ell\ell$'
    signal_label = r'Signal ($m_H=125$ GeV)'

    # *******************
    # general definitions (shouldn't need to change)
    lumi_used = str(HZZAnalysis.lumi*HZZAnalysis.fraction)    
    signal = None
    for s in HZZSamples.samples.keys():
        if s not in HZZAnalysis.stack_order and s!='data': signal = s

    for x_variable,hist in HZZHistograms.hist_dict.items():

        h_bin_width = hist['bin_width']
        h_num_bins = hist['num_bins']
        h_xrange_min = hist['xrange_min']
        h_log_y = hist['log_y']
        h_y_label_x_position = hist['y_label_x_position']
        h_legend_loc = hist['legend_loc']
        h_log_top_margin = hist['log_top_margin'] # to decrease the separation between data and the top of the figure, remove a 0
        h_linear_top_margin = hist['linear_top_margin'] # to decrease the separation between data and the top of the figure, pick a number closer to 1
    
        bins = [h_xrange_min + x*h_bin_width for x in range(h_num_bins+1) ]
        bin_centres = [h_xrange_min+h_bin_width/2 + x*h_bin_width for x in range(h_num_bins) ]

        data_x,_ = np.histogram(data['data'][x_variable].values, bins=bins)
        data_x_errors = np.sqrt(data_x)

        signal_x = None
        if signal_format=='line':
            signal_x,_ = np.histogram(data[signal][x_variable].values,bins=bins,weights=data[signal].totalWeight.values)
        elif signal_format=='hist':
            signal_x = data[signal][x_variable].values
            signal_weights = data[signal].totalWeight.values
            signal_color = HZZSamples.samples[signal]['color']
    
        mc_x = []
        mc_weights = []
        mc_colors = []
        mc_labels = []
        mc_x_tot = np.zeros(len(bin_centres))

        for s in HZZAnalysis.stack_order:
            mc_labels.append(s)
            mc_x.append(data[s][x_variable].values)
            mc_colors.append(HZZSamples.samples[s]['color'])
            mc_weights.append(data[s].totalWeight.values)
            mc_x_heights,_ = np.histogram(data[s][x_variable].values,bins=bins,weights=data[s].totalWeight.values)
            mc_x_tot = np.add(mc_x_tot, mc_x_heights)
    
        mc_x_err = np.sqrt(mc_x_tot)
    
    
        # *************
        # Main plot 
        # *************
        plt.axes([0.1,0.3,0.85,0.65]) #(left, bottom, width, height)
        main_axes = plt.gca()
        main_axes.errorbar( x=bin_centres, y=data_x, yerr=data_x_errors, fmt='ko', label='Data')
        mc_heights = main_axes.hist(mc_x,bins=bins,weights=mc_weights,stacked=True,color=mc_colors, label=mc_labels)
        if Total_SM_label:
            totalSM_handle, = main_axes.step(bins,np.insert(mc_x_tot,0,mc_x_tot[0]),color='black')
        if signal_format=='line':
            main_axes.step(bins,np.insert(signal_x,0,signal_x[0]),color=HZZSamples.samples[signal]['color'], linestyle='--',
                       label=signal)
        elif signal_format=='hist':
            main_axes.hist(signal_x,bins=bins,bottom=mc_x_tot,weights=signal_weights,color=signal_color,label=signal)
        main_axes.bar(bin_centres,2*mc_x_err,bottom=mc_x_tot-mc_x_err,alpha=0.5,color='none',hatch="////",
                  width=h_bin_width, label='Stat. Unc.')
        
        main_axes.set_xlim(left=h_xrange_min,right=bins[-1])
        main_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        main_axes.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=False,right=True,labelright=False)
        if len(labelfile.variable_labels[x_variable].split('['))>1:
            y_units = ' '+labelfile.variable_labels[x_variable][labelfile.variable_labels[x_variable].find("[")+1:labelfile.variable_labels[x_variable].find("]")]
        else: y_units = ''
        main_axes.set_ylabel(r'Events / '+str(h_bin_width)+y_units,fontname='sans-serif',horizontalalignment='right',y=1.0,fontsize=11)
        if h_log_y:
            main_axes.set_yscale('log')
            smallest_contribution = mc_heights[0][0]
            smallest_contribution.sort()
            bottom = smallest_contribution[-2]
            top = np.amax(data_x)*h_log_top_margin
            main_axes.set_ylim(bottom=bottom,top=top)
            main_axes.yaxis.set_major_formatter(CustomTicker())
            locmin = LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            main_axes.yaxis.set_minor_locator(locmin)
        else: 
            main_axes.set_ylim(bottom=0,top=(np.amax(data_x)+math.sqrt(np.amax(data_x)))*h_linear_top_margin)
            main_axes.yaxis.set_minor_locator(AutoMinorLocator())
        
        plt.text(0.05,0.97,'ATLAS',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes,style='italic',weight='bold',fontsize=13)
        plt.text(0.19,0.97,'Open Data',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes,fontsize=13)
        plt.text(0.05,0.9,'for education only',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes,style='italic',fontsize=8)
        plt.text(0.05,0.86,r'$\sqrt{s}=13\,\mathrm{TeV},\;\int L\,dt=$'+lumi_used+'$\,\mathrm{fb}^{-1}$',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes)
        plt.text(0.05,0.78,plot_label,ha="left",va="top",family='sans-serif',transform=main_axes.transAxes)
    
        # Create new legend handles but use the colors from the existing ones 
        handles, labels = main_axes.get_legend_handles_labels()
        if signal_format=='line':
            handles[labels.index(signal)] = Line2D([], [], c=HZZSamples.samples[signal]['color'], linestyle='dashed')
        if Total_SM_label:
            uncertainty_handle = mpatches.Patch(facecolor='none',hatch='////')
            handles.append((totalSM_handle,uncertainty_handle))
            labels.append('Total SM')
    
        # specify order within legend
        new_handles = [handles[labels.index('Data')]]
        new_labels = ['Data']
        for s in reversed(HZZAnalysis.stack_order):
            new_handles.append(handles[labels.index(s)])
            new_labels.append(s)
        if Total_SM_label:
            new_handles.append(handles[labels.index('Total SM')])
            new_labels.append('Total SM')
        else: 
            new_handles.append(handles[labels.index('Stat. Unc.')])
            new_labels.append('Stat. Unc.')
        if signal is not None:
            new_handles.append(handles[labels.index(signal)])
            new_labels.append(signal_label)
        main_axes.legend(handles=new_handles, labels=new_labels, frameon=False, loc=h_legend_loc)
    
    
        # *************
        # Data/MC ratio 
        # *************
        plt.axes([0.1,0.1,0.85,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.errorbar( x=bin_centres, y=data_x/mc_x_tot, yerr=data_x_errors/mc_x_tot, fmt='ko')
        ratio_axes.bar(bin_centres,2*mc_x_err/mc_x_tot,bottom=1-mc_x_err/mc_x_tot,alpha=0.5,color='none',
            hatch="////",width=h_bin_width)
        ratio_axes.plot(bins,np.ones(len(bins)),color='k')
        ratio_axes.set_xlim(left=h_xrange_min,right=bins[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(labelfile.variable_labels[x_variable],fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.set_ylim(bottom=0,top=2.5)
        ratio_axes.set_yticks([0,1,2])
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        if signal is not None:
            ratio_axes.set_ylabel(r'Data/SM',fontname='sans-serif',x=1,fontsize=11)
        else:
            ratio_axes.set_ylabel(r'Data/MC',fontname='sans-serif',fontsize=11)
        
        
        # Generic features for both plots
        main_axes.yaxis.set_label_coords(h_y_label_x_position,1)
        ratio_axes.yaxis.set_label_coords(h_y_label_x_position,0.5)
    
        plt.savefig("HZZ_"+x_variable+".pdf")
    
    return signal_x,mc_x_tot


data_dict = {}
for s in HZZSamples.samples:
    frames = []
    for val in HZZSamples.samples[s]['list']:
        if HZZAnalysis.save_results=='csv': temp = pd.read_csv('resultsHZZ/dataframe_id_'+val+'.csv',index_col='entry')
        elif HZZAnalysis.save_results=='h5':
            if os.path.exists('resultsHZZ/dataframe_id_'+val+'.h5'): temp = pd.read_hdf('resultsHZZ/dataframe_id_'+val+'.h5')
        else:
            print("resultsHZZ/ files don't match save_results! Change save_results in HZZAnalysis.py")
            raise SystemExit
        frames.append(temp)
    data_dict[s] = pd.concat(frames)
plot_data(data_dict)
