import uproot
import pandas as pd
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # for "Total SM & uncertainty" merged legend handle
from matplotlib.lines import Line2D # for dashed line in legend
from matplotlib.ticker import AutoMinorLocator,LogLocator,LogFormatterSciNotation # for minor ticks
import scipy.stats
import os

import HZZSamples
import HZZCuts
import HZZHistograms
import infofile
import labelfile

class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)


save_results = None # 'h5' or 'csv' or None

lumi = 10 # 10 fb-1

fraction = 1 # reduce this is you want the code to run quicker

#tuple_path = "Input/4lep/" # local
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/" # web address

stack_order = [r'$Z,t\bar{t}$','ZZ'] # put smallest contribution first, then increase

parallel = False # True speeds up running, as long as your computer has multiple CPU cores


def expand_columns(df):
    for object_column in df.select_dtypes('object').columns:

        # expand df.object_column into its own dataframe
        object_column_df = df[object_column].apply(pd.Series)

        # rename each variable
        object_column_df = object_column_df.rename(columns = lambda x : object_column + '_' + str(x))

        # join the object_column dataframe back to the original dataframe
        df = pd.concat([df[:], object_column_df[:]], axis=1)
        df = df.drop(object_column,axis=1)

    return df


def read_sample(s):
    print('Processing '+s+' samples')
    frames = []
    for val in HZZSamples.samples[s]['list']:
        prefix = "MC/mc_"
        if s == 'data':
            prefix = "Data/"
        else: prefix += str(infofile.infos[val]["DSID"])+"."
        fileString = tuple_path+prefix+val+".4lep.root" # change ending depending on collection used, e.g. .4lep.root
        if fileString != "":
            temp = read_file(fileString,val)
            if not os.path.exists('resultsHZZ') and save_results!=None: os.makedirs('resultsHZZ')
            if save_results=='csv': temp.to_csv('resultsHZZ/dataframe_id_'+val+'.csv')
            elif save_results=='h5' and len(temp.index)>0:
                temp = expand_columns(temp)
                temp.to_hdf('resultsHZZ/dataframe_id_'+val+'.h5',key='df',mode='w')
            frames.append(temp)
        else:
            print("Error: "+val+" not found!")
    data_s = pd.concat(frames)
    return data_s


def get_data_from_files():

    data = {}
    for s in HZZSamples.samples:
        data[s] = read_sample(s)
    
    return data


def calc_weight(mcWeight,scaleFactor_PILEUP,scaleFactor_ELE,
                scaleFactor_MUON, scaleFactor_LepTRIGGER):
    return mcWeight*scaleFactor_PILEUP*scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER

def get_xsec_weight(totalWeight,sample):
    info = infofile.infos[sample]
    weight = (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"]) #*1000 to go from fb-1 to pb-1
    weight *= totalWeight
    return weight

def calc_mllll(lep_pts,lep_etas,lep_phis):
    theta_0 = 2*math.atan(math.exp(-lep_etas[0]))
    theta_1 = 2*math.atan(math.exp(-lep_etas[1]))
    theta_2 = 2*math.atan(math.exp(-lep_etas[2]))
    theta_3 = 2*math.atan(math.exp(-lep_etas[3]))
    p_0 = lep_pts[0]/math.sin(theta_0)
    p_1 = lep_pts[1]/math.sin(theta_1)
    p_2 = lep_pts[2]/math.sin(theta_2)
    p_3 = lep_pts[3]/math.sin(theta_3)
    pz_0 = p_0*math.cos(theta_0)
    pz_1 = p_1*math.cos(theta_1)
    pz_2 = p_2*math.cos(theta_2)
    pz_3 = p_3*math.cos(theta_3)
    px_0 = p_0*math.sin(theta_0)*math.cos(lep_phis[0])
    px_1 = p_1*math.sin(theta_1)*math.cos(lep_phis[1])
    px_2 = p_2*math.sin(theta_2)*math.cos(lep_phis[2])
    px_3 = p_3*math.sin(theta_3)*math.cos(lep_phis[3])
    py_0 = p_0*math.sin(theta_0)*math.sin(lep_phis[0])
    py_1 = p_1*math.sin(theta_1)*math.sin(lep_phis[1])
    py_2 = p_2*math.sin(theta_2)*math.sin(lep_phis[2])
    py_3 = p_3*math.sin(theta_3)*math.sin(lep_phis[3])
    sumpz = pz_0 + pz_1 + pz_2 + pz_3
    sumpx = px_0 + px_1 + px_2 + px_3
    sumpy = py_0 + py_1 + py_2 + py_3
    sumE = p_0 + p_1 + p_2 + p_3
    mllll = sumE**2 - sumpz**2 - sumpx**2 - sumpy**2
    return math.sqrt(mllll)/1000 #/1000 to go from MeV to GeV


def read_file(path,sample):
    start = time.time()
    print("\tProcessing: "+sample+" file")
    data_all = pd.DataFrame()
    mc = uproot.open(path)["mini"]
    numevents = uproot.numentries(path, "mini")
    for data in mc.iterate(["lep_n","lep_pt","lep_eta","lep_phi","lep_E","lep_charge","lep_type","lep_ptcone30","lep_etcone20",
                         "mcWeight","scaleFactor_PILEUP","scaleFactor_ELE","scaleFactor_MUON",
                                                     "scaleFactor_LepTRIGGER"], flatten=False, entrysteps=2500000, outputtype=pd.DataFrame, entrystop=numevents*fraction):

        nIn = len(data.index)

        if 'data' not in sample:
            data['totalWeight'] = np.vectorize(calc_weight)(data.mcWeight,data.scaleFactor_PILEUP,data.scaleFactor_ELE,data.scaleFactor_MUON,data.scaleFactor_LepTRIGGER)    
            data['totalWeight'] = np.vectorize(get_xsec_weight)(data.totalWeight,sample)

        data.drop(["mcWeight","scaleFactor_PILEUP","scaleFactor_ELE","scaleFactor_MUON","scaleFactor_LepTRIGGER"], axis=1, inplace=True)

        # cut on number of leptons
        fail =data[ np.vectorize(HZZCuts.cut_n_lep)(data.lep_n)].index
        data.drop(fail, inplace=True)
    
        # cut on lepton charge
        fail = data[ np.vectorize(HZZCuts.cut_lep_charge)(data.lep_charge) ].index
        data.drop(fail, inplace=True)
    
        #cut on the transverse momentum of the leptons
        fail =data[ np.vectorize(HZZCuts.cut_lep_pt_012)(data.lep_pt)].index
        data.drop(fail,inplace=True)
    
        # cut on lepton type
        fail = data[ np.vectorize(HZZCuts.cut_lep_type)(data.lep_type) ].index
        data.drop(fail, inplace=True)

        # cut on lepton momentum isolation
        #fail = data[ np.vectorize(HZZCuts.cut_lep_ptcone)(data.lep_ptcone30,data.lep_pt) ].index
        #data.drop(fail, inplace=True)

        # cut on lepton energy isolation
        #fail = data[ np.vectorize(HZZCuts.cut_lep_etcone)(data.lep_etcone20,data.lep_pt) ].index
        #data.drop(fail, inplace=True)

        data['mllll'] = np.vectorize(calc_mllll)(data.lep_pt,data.lep_eta,data.lep_phi)
        
        nOut = len(data.index)
        data_all = data_all.append(data)
        elapsed = time.time() - start
        print("\t\t"+sample+" time taken: "+str(elapsed)+"s, nIn: "+str(nIn)+", nOut: "+str(nOut))

    return data_all


def plot_data(data):

    signal_format = 'hist' # 'line' for line above SM stack
                           # 'hist' for bar above SM stack
                           # None for signal as part of SM stack
    Total_SM_label = False # for Total SM black line in plot and legend
    plot_label = r'$H \rightarrow ZZ^* \rightarrow \ell\ell\ell\ell$'
    signal_label = r'Signal ($m_H=125$ GeV)' # r''

    # *******************
    # general definitions (shouldn't need to change)
    lumi_used = str(lumi*fraction)    
    signal = None
    for s in HZZSamples.samples.keys():
        if s not in stack_order and s!='data': signal = s

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

        for s in stack_order:
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
        plt.clf()
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
        for s in reversed(stack_order):
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


if __name__=="__main__":
    start = time.time()
    data = get_data_from_files()
    signal_yields,background_yields = plot_data(data)
    elapsed = time.time() - start
    print("Time taken: "+str(elapsed))
