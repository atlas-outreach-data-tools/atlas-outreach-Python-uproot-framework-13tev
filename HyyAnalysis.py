import uproot3 as uproot
import pandas as pd
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import PolynomialModel, GaussianModel
import matplotlib.patches as mpatches # for "Total SM & uncertainty" merged legend handle
from matplotlib.lines import Line2D # for dashed line in legend
from matplotlib.ticker import MaxNLocator,AutoMinorLocator,LogLocator,LogFormatterSciNotation # for minor ticks
import scipy.stats
import os

import HyySamples
import HyyCuts
import HyyHistograms


class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)


save_results = None # 'h5' or 'csv' or None

lumi = 0.5 # fb-1 for data_A only
#lumi = 10 # 10 fb-1 for data_A,B,C,D

fraction = 1 # reduce this is you want the code to run quicker

#tuple_path = "Input/GamGam/" # local
tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/" # web address

stack_order = [] # put smallest contribution first, then increase


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
    for val in HyySamples.samples[s]['list']:
        fileString = tuple_path+"Data/"+val+".GamGam.root" # change ending depending on collection used, e.g. .4lep.root
        if fileString != "":
            temp = read_file(fileString,val)
            if not os.path.exists('resultsHyy') and save_results!=None: os.makedirs('resultsHyy')
            if save_results=='csv': temp.to_csv('resultsHyy/dataframe_id_'+val+'.csv')
            elif save_results=='h5' and len(temp.index)>0:
                temp = expand_columns(temp)
                temp.to_hdf('resultsHyy/dataframe_id_'+val+'.h5',key='df',mode='w')
            frames.append(temp)
        else:
            print("Error: "+val+" not found!")
    data_s = pd.concat(frames)
    return data_s


def get_data_from_files():

    data = {}
    for s in HyySamples.samples:
        data[s] = read_sample(s)
    
    return data


def calc_myy(photon_pt,photon_eta,photon_phi):
    myy = 2*photon_pt[0]*photon_pt[1]
    cosh = math.cosh(photon_eta[0]-photon_eta[1])
    cos = math.cos(photon_phi[0]-photon_phi[1])
    myy *= ( cosh - cos )
    return math.sqrt(myy)/1000 #/1000 to go from MeV to GeV


def read_file(path,sample):
    start = time.time()
    print("\tProcessing: "+sample)
    data_all = pd.DataFrame()
    mc = uproot.open(path)["mini"]
    numevents = uproot.numentries(path, "mini")
    for data in mc.iterate(["photon_n","photon_pt","photon_eta","photon_phi","photon_etcone20","photon_ptcone30",
                                                     "photon_isTightID"], flatten=False, entrysteps=2500000, outputtype=pd.DataFrame, entrystop=numevents*fraction):

        nIn = len(data.index)


        # Calculate reconstructed diphoton invariant mass
        data['myy'] = np.vectorize(calc_myy)(data.photon_pt,data.photon_eta,data.photon_phi)
    
        # Cut on number of photons
        fail = data[ np.vectorize(HyyCuts.cut_photon_n)(data.photon_n)].index
        data.drop(fail, inplace=True)

        # Cut on pseudorapidity outside fiducial region
        fail = data[ np.vectorize(HyyCuts.cut_photon_eta_fiducial)(data.photon_eta)].index
        data.drop(fail, inplace=True)

        # Cut on pseudorapidity inside barrel/end-cap transition region
        fail = data[ np.vectorize(HyyCuts.cut_photon_eta_transition)(data.photon_eta)].index
        data.drop(fail, inplace=True)

        # Cut on transverse momentum of the photons
        fail = data[ np.vectorize(HyyCuts.cut_photon_pt)(data.photon_pt)].index
        data.drop(fail, inplace=True)

        # Cut on photon reconstruction
        fail = data[ np.vectorize(HyyCuts.cut_photon_reconstruction)(data.photon_isTightID)].index
        data.drop(fail, inplace=True)

        # Cut on energy isolation
        fail = data[ np.vectorize(HyyCuts.cut_isolation_et)(data.photon_etcone20)].index
        data.drop(fail, inplace=True)

        # Cut on lower limit of reconstructed invariant mass
        fail = data[ np.vectorize(HyyCuts.cut_mass_lower)(data.myy)].index
        data.drop(fail, inplace=True)

        # Cut on upper limit of reconsructed invariant mass
        fail = data[ np.vectorize(HyyCuts.cut_mass_upper)(data.myy)].index
        data.drop(fail, inplace=True)
    
        # dataframe contents can be printed at any stage like this
        #print(data)

        # dataframe column can be printed at any stage like this
        #print(data['photon_pt'])

        # dataframe columns can be printed at any stage like this
        #print(data[['photon_pt','photon_eta']])
        
        nOut = len(data.index)
        data_all = data_all.append(data)
        elapsed = time.time() - start
        print("\t\tTime taken: "+str(elapsed)+", nIn: "+str(nIn)+", nOut: "+str(nOut))

    return data_all


def plot_data(data):
    
    signal_format = None # 'line' or 'hist' or None
    Total_SM_label = False # for Total SM black line in plot and legend
    plot_label = r'$H \rightarrow \gamma\gamma$'
    signal_label = ''
    
    # *******************
    # general definitions (shouldn't need to change)
    lumi_used = str(lumi*fraction)    
    signal = None
    for s in HyySamples.samples.keys():
        if s not in stack_order and s!='data': signal = s

    for x_variable,hist in HyyHistograms.hist_dict.items():

        h_bin_width = hist['bin_width']
        h_num_bins = hist['num_bins']
        h_xrange_min = hist['xrange_min']
        h_xlabel = hist['xlabel']
        h_log_y = hist['log_y']
        h_y_label_x_position = hist['y_label_x_position']
        h_legend_loc = hist['legend_loc']
        h_log_top_margin = hist['log_top_margin'] # to decrease the separation between data and the top of the figure, remove a 0
        h_linear_top_margin = hist['linear_top_margin'] # to decrease the separation between data and the top of the figure, pick a number closer to 1

        bins = [h_xrange_min + x*h_bin_width for x in range(h_num_bins+1) ]
        bin_centres = [h_xrange_min+h_bin_width/2 + x*h_bin_width for x in range(h_num_bins) ]

        data_x,_ = np.histogram(data['data'][x_variable].values, bins=bins)
        data_x_errors = np.sqrt(data_x)
    
        # data fit
        polynomial_mod = PolynomialModel(4)
        gaussian_mod = GaussianModel()
        bin_centres_array = np.asarray(bin_centres)
        pars = polynomial_mod.guess(data_x, x=bin_centres_array, c0=data_x.max(), c1=0, c2=0, c3=0, c4=0)
        pars += gaussian_mod.guess(data_x, x=bin_centres_array, amplitude=91.7, center=125., sigma=2.4)
        model = polynomial_mod + gaussian_mod
        out = model.fit(data_x, pars, x=bin_centres_array, weights=1/data_x_errors)
    
        # background part of fit
        params_dict = out.params.valuesdict()
        c0 = params_dict['c0']
        c1 = params_dict['c1']
        c2 = params_dict['c2']
        c3 = params_dict['c3']
        c4 = params_dict['c4']
        background = c0 + c1*bin_centres_array + c2*bin_centres_array**2 + c3*bin_centres_array**3 + c4*bin_centres_array**4

        signal_x = None
        if signal_format=='line':
            signal_x,_ = np.histogram(data[signal][x_variable].values,bins=bins,weights=data[signal].totalWeight.values)
        elif signal_format=='hist':
            signal_x = data[signal][x_variable].values
            signal_weights = data[signal].totalWeight.values
            signal_color = HyySamples.samples[signal]['color']
        signal_x = data_x - background
    
        mc_x = []
        mc_weights = []
        mc_colors = []
        mc_labels = []
        mc_x_tot = np.zeros(len(bin_centres))

        for s in stack_order:
            mc_labels.append(s)
            mc_x.append(data[s][x_variable].values)
            mc_colors.append(HyySamples.samples[s]['color'])
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
        if Total_SM_label:
            totalSM_handle, = main_axes.step(bins,np.insert(mc_x_tot,0,mc_x_tot[0]),color='black')
        if signal_format=='line':
            main_axes.step(bins,np.insert(signal_x,0,signal_x[0]),color=HyySamples.samples[signal]['color'], linestyle='--',
                       label=signal)
        elif signal_format=='hist':
            main_axes.hist(signal_x,bins=bins,bottom=mc_x_tot,weights=signal_weights,color=signal_color,label=signal)
        main_axes.bar(bin_centres,2*mc_x_err,bottom=mc_x_tot-mc_x_err,alpha=0.5,color='none',hatch="////",
                  width=h_bin_width, label='Stat. Unc.')
        main_axes.plot(bin_centres, out.best_fit, '-r', label='Sig+Bkg Fit ($m_H=125$ GeV)')
        main_axes.plot(bin_centres, background, '--r', label='Bkg (4th order polynomial)')
        
        main_axes.set_xlim(left=h_xrange_min,right=bins[-1])
        main_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        main_axes.tick_params(which='both',direction='in',top=True,labeltop=False,labelbottom=False,right=True,labelright=False)
        if len(h_xlabel.split('['))>1:
            y_units = ' '+h_xlabel[h_xlabel.find("[")+1:h_xlabel.find("]")]
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
            main_axes.yaxis.get_major_ticks()[0].set_visible(False)
        
        plt.text(0.2,0.97,'ATLAS Open Data',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes,fontsize=13)
        plt.text(0.2,0.9,'for education',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes,style='italic',fontsize=8)
        plt.text(0.2,0.86,r'$\sqrt{s}=13\,\mathrm{TeV},\;\int L\,dt=$'+lumi_used+'$\,\mathrm{fb}^{-1}$',ha="left",va="top",family='sans-serif',transform=main_axes.transAxes)
        plt.text(0.2,0.78,plot_label,ha="left",va="top",family='sans-serif',transform=main_axes.transAxes)
    
        # Create new legend handles but use the colors from the existing ones 
        handles, labels = main_axes.get_legend_handles_labels()
        if signal_format=='line':
            handles[labels.index(signal)] = Line2D([], [], c=HyySamples.samples[signal]['color'], linestyle='dashed')
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
            new_handles.append(handles[labels.index('Sig+Bkg Fit ($m_H=125$ GeV)')])
            new_handles.append(handles[labels.index('Bkg (4th order polynomial)')])
            new_labels.append('Sig+Bkg Fit ($m_H=125$ GeV)')
            new_labels.append('Bkg (4th order polynomial)')
        if signal is not None:
            new_handles.append(handles[labels.index(signal)])
            new_labels.append(signal_label)
        main_axes.legend(handles=new_handles, labels=new_labels, frameon=False, loc=h_legend_loc)
    
    
        # *************
        # Data-Bkg plot 
        # *************
        plt.axes([0.1,0.1,0.85,0.2]) #(left, bottom, width, height)
        ratio_axes = plt.gca()
        ratio_axes.yaxis.set_major_locator(MaxNLocator(nbins='auto',symmetric=True))
        ratio_axes.errorbar( x=bin_centres, y=signal_x, yerr=data_x_errors, fmt='ko')
        ratio_axes.plot(bin_centres, out.best_fit-background, '-r')
        ratio_axes.plot(bin_centres, background-background, '--r') 
        ratio_axes.set_xlim(left=h_xrange_min,right=bins[-1])
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator()) # separation of x axis minor ticks
        ratio_axes.xaxis.set_label_coords(0.9,-0.2) # (x,y) of x axis label # 0.2 down from x axis
        ratio_axes.set_xlabel(h_xlabel,fontname='sans-serif',fontsize=11)
        ratio_axes.tick_params(which='both',direction='in',top=True,labeltop=False,right=True,labelright=False)
        ratio_axes.yaxis.set_minor_locator(AutoMinorLocator())
        if signal_format=='line' or signal_format=='hist':
            ratio_axes.set_ylabel(r'Data/SM',fontname='sans-serif',x=1,fontsize=11)
        else:
            ratio_axes.set_ylabel(r'Events-Bkg',fontname='sans-serif',x=1,fontsize=11)
        
        
        # Generic features for both plots
        main_axes.yaxis.set_label_coords(h_y_label_x_position,1)
        ratio_axes.yaxis.set_label_coords(h_y_label_x_position,0.5)
    
        plt.savefig("Hyy_"+x_variable+".pdf",bbox_inches='tight')
    
        print('chi^2 = '+str(out.chisqr))
        print('gaussian centre = '+str(params_dict['center']))
        print('gaussian sigma = '+str(params_dict['sigma']))
        print('gaussian fwhm = '+str(params_dict['fwhm']))
    
    return signal_x,mc_x_tot

if __name__=="__main__":
    start = time.time()
    data = get_data_from_files()
    signal_yields,background_yields = plot_data(data)
    elapsed = time.time() - start
    print("Time taken: "+str(elapsed))
