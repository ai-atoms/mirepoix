import os, shutil
import numpy as np

import pickle
import csv, h5py
from h5py import Group, File

from typing import List, Dict, TypedDict
from difflib import SequenceMatcher
from ase import Atoms

class DataDfct(TypedDict):
    """Little class associated to the internal structure of ```DfctAnalysisStorage```"""
    # Point defects and Nanophases
    center : np.ndarray
    nb_defect : int

    # dislocation
    dislocation_center : np.ndarray
    burgers_vector_type : str
    burger_vector : np.ndarray
    dislocation_size : float
    dislocation_lenght : float
    dislocation_density : float

class DfctAnalysisStorage : 
    def __init__(self, path_writing : os.PathLike[str] = './dfct_analysis.pkl') :
        """Storage object to export data from analysis to systematic format as ```.csv``` or ```.h5```
        
        Parameters
        ----------

        path_writing : os.PathLike[str]
            Path to the pickle file containing all data from defect analysis
        """
        self.path_writing = path_writing
        self.data : Dict[str, Dict[str, Dict[int, DataDfct]]] = {}
        self.data_distribution : Dict[str, Dict[str, float | np.ndarray]] = {}

    def update_distribution(self, name : str,
                            key_data : str,
                            data : float | np.ndarray) -> None :
        """Update ```data_distribution`` dictionnary for global defect population analysis
        
        Parameters
        ----------

        name : str
            Key of the global configuration to update

        key_data : str
            Key of data to update (C15, A15, partial dislo, total_dislo etc.)

        data : float | np.ndarray
            Associated data to update 
        
        """
        if name in self.data_distribution.keys() :
            self.data_distribution[name].update({key_data:data})
        else :
            self.data_distribution[name] = {key_data:data}
        return 

    def update_data(self, name : str, 
                    dictionnary : Dict[str, Dict[int, DataDfct]]) -> None :
        """Update ```data``` dictionnary for cluster dynamics
        
        Parameters
        ----------

        name : str
            Key of the global configuration to update

        dictionnary : Dict[str, Dict[int, DataDfct]]
            Standartised dictionnary used to be exported as .csv or .h5

        """
        if name in self.data.keys() :
            self.data[name].update(dictionnary)
        else :
            self.data[name] = dictionnary
        return
    
    def add_or_update_dfct_group(self, hdf5_group : Group,
                            dfct : str,
                            index : int, 
                            data_dfct : DataDfct) -> None :
        """Update the .h5 file 
        
        Parameters
        ----------

        hdf5_group : ```Group```
            ```Group``` associated to a given configuration

        dfct : str 
            Type of defect to update

        index : int 
            Cluster index to update

        data_dfct : ```DataDfct```
            ```DataDfct``` dictionnary associated to (dfct, index)
        """

        if dfct not in hdf5_group:
            # Create group for dfct
            dfct_group = hdf5_group.create_group(str(index))
            index_group = dfct_group.create_group(dfct)

            for key, val in data_dfct : 
                index_group.create_dataset(key, data=val)

        else : 
            dfct_group = hdf5_group[dfct]
            index_group = dfct_group.create_group(dfct)

            for key, val in data_dfct : 
                index_group.create_dataset(key, data=val)
            
        return 

    def export_to_csv(self, path_csv : os.PathLike[str]) -> None :
        """Export the whole defect data from analysis to .csv files
        One file is created for a given configuration and a given type of defect

        Parameters
        ----------

        path_csv : os.PathLike[str]
            Path to the directory to write .csv files

        """

        if os.path.exists(path_csv) : 
            shutil.rmtree(path_csv)
        os.mkdir(path_csv)

        for name, dict in self.data.items() : 
            for dfct, sub_dict in dict.items() : 
                if len(sub_dict) == 0 : 
                    continue
                else :
                    path_csv = f'{path_csv}/{name}_{dfct}.csv'
                
                with open(path_csv, 'w') as csv_file:  
                    writer = csv.writer(csv_file)
                    for key, value in sub_dict.items():
                       writer.writerow([key, value])
        return 

    def export_to_h5(self, path_h5 : os.PathLike[str]) -> None : 
        """Export the whole defect data from analysis to .h5 files 
        Create an unique .h5 file containing nested dictionnary

        Parameters
        ----------

        path_h5 : os.PathLike[str]
            Path to the .h5 file to write 

        """
        if os.path.exists(path_h5) : 
            shutil.rmtree(path_h5)

        h5_file = h5py.File(path_h5, 'w')
        for name, dict in self.data.items() : 
            # group for configuration
            name_group = h5_file.create_group(name)
            for dfct, sub_dict in dict.items() :
                if len(sub_dict) == 0 :
                    continue
                else :
                    for idx, data in sub_dict.items() : 
                        # add group for dfct and associated clusters
                        self.add_or_update_dfct_group(name_group,
                                                      dfct,
                                                      idx,
                                                      data)
                        
        return 

    def write_pkl(self) -> None : 
        """Write pickle file for ```DfctAnalysisStorage``` object
        
        Parameters
        ----------

        path_writing : os.PathLike[str]
            Writing path for pickle file
        """
        pickle.dump(self, open(self.path_writing,'wb'))
        return
