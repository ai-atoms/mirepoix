import os, sys
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
# mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Dict, Tuple

sys.path.append(os.getcwd())
sys.path.insert(0,'../../')
from dfct_storage import DfctAnalysisStorage

# -- camilofs
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
import scienceplots

plt.style.use(['science'])

#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fontsize = 16

class PlotDistribution : 
    """ Analysis class for comparison beteween nanophases populations and dislocation populations"""
    def __init__(self, path_pkl : os.PathLike[str],
                 dose_max : float,
                 mode_grid_dose : str = 'uniform') :
        """Init method for ```PlotDistribution``` 
        
        path_pkl : os.PathLike[str]
            Path to the pickle file from ```DfctStorageAnalysis``` containing whole data for analysis

        dose_max : float
            Maximal dose applied during the simulatin (in dpa)

        mode_grid_dose : str
            Naive mode to evaluate dose for each configuration (should be changed in future !)
        """
        self.path_pkl = path_pkl
        self.pkl_data = self.load_pkl()
        self.dose_max = dose_max
        self.array_dose = self.build_dose_grid(mode=mode_grid_dose)
        self.size = len(self.pkl_data.data_distribution)

    def get_dislocation_keys(self) -> List[str] : 
        """Extract all dislocation keys contain in pkl file
        
        Returns
        -------

        List[str]
            List of dislocation key in pkl file
        """
        distribution_dic = self.pkl_data.data_distribution
        first_key = next(iter(distribution_dic))
        return [key for key in distribution_dic[first_key].keys() if 'dislo' in key]

    def build_dose_grid(self, mode : str = 'uniform') -> np.ndarray : 
        """Naive method to compute dose associated to each configuration
        
        Parameters
        ----------

        mode : str
            Type of grid used for dose calcualtion

        Returns
        -------

        np.ndarray 
            Array of doses associated to each configurations in pkl file
        
        """
        implemented_mode = ['uniform']
        dic_method = {'uniform': lambda nb, max : np.linspace(0.0,max, num=nb)}
        
        if mode not in implemented_mode : 
            raise NotImplementedError(f'This mode is not implemented {mode}')
        
        return dic_method[mode](len(self.pkl_data.data_distribution),self.dose_max)


    def load_pkl(self) -> DfctAnalysisStorage : 
        """Load method to extract data from pickle
        
        Returns 
        -------

        ```DfctAnalysisStorage``` 
            ```DfctAnalysisStorage``` associated to the pkl file
        """
        return pickle.load(open(self.path_pkl,'rb'))
    

    def plot_dislocation_density(self, axis : Axes,
                                 log_x : bool = False) -> Axes :
        """Plot dislocation densities depending on dose 
        
        Parameters
        ----------

        axis : Axes 
            Matplotlib axes to plot on

        log_x : bool
            x axis is turned to logscale if True 

        Returns
        -------

        Axes 
            Updated Axes object

        """

        # to be updated !
        dic_equiv = {'<100>_dislo':{'color':'royalblue',
                              'name':r'$\langle 100 \rangle$ dislocation density'},
                     '<111>_dislo':{'color':'forestgreen',
                                    'name':r'$\frac{1}{2} \langle 111 \rangle$ dislocation density'},
                     'total_dislo':{'color':'grey',
                                    'name':r'Total dislocation density'}}

        convertion_factor = 1.0e20
        key_dislo = self.get_dislocation_keys()
        data_distribution = self.pkl_data.data_distribution
        print([(key, len(val['C15'])) for key, val in data_distribution.items()])
        # exit(0)

        for k_d in key_dislo : 
            array_data_k_d = np.array([data_distribution[key][k_d] for key in data_distribution.keys()])
            axis.plot(self.array_dose, 
                      array_data_k_d*convertion_factor,
                      color=dic_equiv[k_d]['color'],
                      label=dic_equiv[k_d]['name'])
        
        if log_x : 
            axis.set_xscale('log')

        axis.tick_params(axis='x', labelsize=15)
        axis.tick_params(axis='y', labelsize=15)

        # dose
        majorFormatter = mpl.ticker.FormatStrFormatter(r'%1.2f')
        majorLocator = mpl.ticker.MultipleLocator(0.02)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        axis.xaxis.set_major_locator(majorLocator)
        axis.xaxis.set_major_formatter(majorFormatter)
        axis.xaxis.set_minor_locator(minorLocator)

        # stack overflow <3
        def MyFormatter(x,lim):
            if x == 0 :
                return r"0.0"
            
            else : 
                val = x/(1e16)
                return r"{:1.1f}".format(val)
        
        majorFormatter = FuncFormatter(MyFormatter)
        majorLocator = mpl.ticker.MultipleLocator(2.0e15)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        axis.yaxis.set_major_locator(majorLocator)
        axis.yaxis.set_major_formatter(majorFormatter)
        axis.yaxis.set_minor_locator(minorLocator)
        axis.tick_params(which='both', width=1,labelsize=15)
        axis.tick_params(which='major', length=8)
        axis.tick_params(which='minor', length=4, color='black')


        axis.set_xlim(0.0, self.dose_max)
        #axis.set_xlabel(r'Dose (dpa)', fontsize=fontsize)
        axis.set_ylabel(r'Dislocation density ($10^{16} \: m^{-2}$)', fontsize=fontsize)
        axis.legend(frameon=False,
                    fontsize=14)
        return axis
    
    
    def plot_nanophase_density(self, axis : Axes,
                               name_nano : str,
                               colormap : str = 'viridis',
                               log_x : bool = False,
                               nb_bin_y : int = 22) -> Axes :
        """Plot nanophases size histograms depending on dose
        
        Parameters
        ----------

        axis : Axes 
            Matplotlib axes to plot on

        name_nano : str
            Name of the associated nanophase

        colormap : str 
            colormap used to plot 2d histogram

        log_x : bool
            x axis is turned to logscale if True 

        nb_bin_y : int 
            Number of bins used for y axis (nanophase cluster size)

        Returns
        -------

        Axes 
            Updated Axes object

        """ 
        data_distribution = self.pkl_data.data_distribution
        array_dose = []
        array_nano = []
        for id, key in enumerate(data_distribution.keys()) :
            dose = self.array_dose[id]
            data_nano = data_distribution[key][name_nano]

            array_dose += [dose for k in range(len(data_nano))]
            array_nano += data_nano.tolist()
        
        # axis.tick_params(axis='x', labelsize=16)
        # axis.tick_params(axis='y', labelsize=16)

        hist = axis.hist2d(array_dose, 
                           array_nano, 
                           bins=[int(len(data_distribution)*0.8), nb_bin_y], 
                           cmap=colormap, 
                           cmin=1,
                           cmax=40) 

        if log_x : 
            axis.set_xscale('log')

        # dose
        majorFormatter = mpl.ticker.FormatStrFormatter(r'%1.2f')
        majorLocator = mpl.ticker.MultipleLocator(0.02)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        axis.xaxis.set_major_locator(majorLocator)
        axis.xaxis.set_major_formatter(majorFormatter)
        axis.xaxis.set_minor_locator(minorLocator)


        majorFormatter = mpl.ticker.FormatStrFormatter(r'%3d')
        majorLocator = mpl.ticker.MultipleLocator(10)
        minorLocator = mpl.ticker.AutoMinorLocator(2)
        axis.yaxis.set_major_locator(majorLocator)
        axis.yaxis.set_major_formatter(majorFormatter)
        axis.yaxis.set_minor_locator(minorLocator)

        axis.tick_params(which='both', width=1,labelsize=15)
        axis.tick_params(which='major', length=8)
        axis.tick_params(which='minor', length=4, color='black')
        
        # axis.set_xlim(0.0, self.dose_max)
        axis.set_xlim(0.0, 0.08)
        axis.set_ylim(0, 50)
        axis.set_xlabel(r"Dose (dpa)", fontsize=fontsize)
        axis.set_ylabel(r"%s Size (atoms)"%(name_nano), fontsize=fontsize)
        return axis, hist[3]
    
    def plot_distribution(self, nano_phase : str,
                          colormap : str = 'viridis',
                          log_x : bool = False) -> None : 
        """General method to plot nanophase size distribution vs. dislocation densities depending on dose
        
        Parameters
        ----------

        nano_phase : str
            Name of the nanophase considered

        colormap : str
            Colormap used for 2d histogram

        log_x : bool 
            x axis is turned to logscale if True 
        """    
    
        fig, axis = plt.subplots(nrows=2, ncols=1, figsize=(7.6,9),
                                 sharex=True, layout='constrained')   
        self.plot_dislocation_density(axis[0], 
                                      log_x=log_x)
        _, array = self.plot_nanophase_density(axis[1],
                                    nano_phase, 
                                    colormap=colormap,
                                    log_x=log_x)

        cbar = fig.colorbar(array, ax=axis[1], pad=0.0)
        cbar.ax.tick_params(labelsize=15)  # Increase tick label font size

        plt.savefig(f'distribution_{nano_phase}_dislo.png', dpi=300)
        plt.savefig(f'distribution_{nano_phase}_dislo.pdf', dpi=300)
        # plt.show()
        return 
    
########################################
### INPUTS
########################################
path_data = 'data/fpa70/descriptor/fpa70_distribution.pkl'
#######################################

obj_plot = PlotDistribution(path_data,
                            0.088,
                            mode_grid_dose='uniform')
obj_plot.plot_distribution('C15',
                           colormap='viridis',
                           log_x=False)
