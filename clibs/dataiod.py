import os
from .tensormultiprocessings import RayTorchMP, cpuworkers, ResourceManager, raydeclare
import torch
import seaborn as sns
import pandas as pd
import traceback
import pickle
from .timegen import Curtimeout, Starttime
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt, font_manager as fm
from matplotlib import font_manager as fm
from matplotlib.ft2font import FT2Font
import multiprocessing
from pathlib import Path


def search_ucfontKR():
    if os.name == 'nt':  # Check if the operating system is Windows
        froot=Path(os.environ["windir"], "Fonts")
    else:
        froot=None
    matched_fonts={}#initialize matched fonts variable
    
    for file in froot.rglob(r"*.ttf|*.otf|*.ttc"):
        try:
            name=fm.FontProperties().get_name()#get font name
            fontpath=file
            font=FT2Font(fontpath)#get font object
            if all(font.get_char_index(ord(ch))>0 for ch in '한글문자를출력하는지테스트'):  # Check if the font supports Korean characters
                matched_fonts[name]=fontpath#store matched font name and path
        except Exception:
            continue
        if not matched_fonts:
            raise ValueError("No Korean font found in the system.")
        print(f"{matched_fonts[0]} 폰트가 한글을 지원합니다. 경로: {matched_fonts[1]}")
        plt.rcParams["font.family"]=matched_fonts[0]#set the first matched font as the default font for the plots
        plt.rcParams["axes.unicode_minus"]=False#to avoid minus sign rendering issues in plots
        plt.rcParams["font.size"]=12#setting font size for the plots
        plt.rcParams["figure.figsize"]=(10, 6)#setting figure size for the plots
        plt.rcParams["axes.grid"]=True#setting grid for the plots
        plt.rcParams["axes.facecolor"]='white'#setting background color for the plots
        plt.rcParams["axes.edgecolor"]='black'#setting edge color for the plots
        plt.rcParams["axes.labelcolor"]='black'#setting label color for the plots
        plt.rcParams["axes.titlesize"]=16#setting title size for the plots
        plt.rcParams["axes.titleweight"]='bold'#setting title weight for the plots
        plt.rcParams["axes.labelsize"]=14#setting label size for the plots
        plt.rcParams["axes.labelweight"]='bold'#setting label weight for the plots
    return matched_fonts
    


class Pathbuild():
    def __init__(self, pathbyruntime):
        """_"relative pathbuilder can include times"_

        Args:
            pathbyruntime (_Bool_): _"True: use"_
        """
        curtimepath=Curtimeout().str
        self.curtimepath = curtimepath if pathbyruntime else Starttime.str
        
    def stOrtfpathbuild(self, subfolderlist, filename, filetype):
        """do not call this method directly out of the class!"""
        #subfolds=[]
        #filenames=[]
        #filetypes=[]
        #for ind, var in enumerate(subfolderlist):
        subfolderlist=f"{subfolderlist}+{os.sep}"
        #for ind, var in enumerate(subfolderlist):
            #filenames[ind]=filename[ind]
        #for ind, var in enumerate(filetype):
            #filetypes[ind]=os.extsep+filetype[ind]
        res=f"{os.curdir}+{os.sep}+{subfolderlist}+{filename}+{os.extsep}+{filetype}"
        #res=[os.curdir+os.sep+f"{subfolds[ind]}"+f"{filenames[ind2]}"+os.extsep+f"{filetypes[ind3]}" for ind, ind2, ind3 in range(0, len(subfolds), 1),  range(0, len(filenames), 1)),  range(0, len(filetypes, 1)]
        return res
    
    def __listORstr(self, pathlist, additionals=""):
        if not isinstance(pathlist, str):
            res=[f"{additionals}"+f"{pathlist[ind]}" for ind in range(0, len(pathlist), 1) ]
        else:
            res=f"{additionals+pathlist}"
        return res
    
    def rtfpathbuild(self, subfolderlist, filename, filetype):
        """_"multifile/folder pathbuilder"_

        Args:
            subfolderlist (_"list of str"_): _targetsubfolders_
            filename (_"list of str"_): _targetfilenames_
            filetype (_"list of str"_): _targetextensions_

        Returns:
            _"list of str"_: _"path list"_
        """
        subfolderlist=self.__listORstr(subfolderlist)
        filename=self.__listORstr(filename)
        filetype=self.__listORstr(filetype)
        
        return self.stOrtfpathbuild(subfolderlist, filename, filetype)
        
    def rtfpath_time(self, subfolderlist, filename, filetype):
        """_"runtime included multifile/folder pathbuilder"_

        Args:
            subfolderlist (_"list of str"_): _targetsubfolders_
            filename (_"list of str"_): _targetfilenames_
            filetype (_"list of str"_): _targetextensions_

        Returns:
            _"list of str"_: _"path list, runtime included"_
        """
        subfoldertlist=self.__listORstr(subfolderlist, self.curtimepath)
        filename=self.__listORstr(filename, self.curtimepath)
        filetype=self.__listORstr(filetype)
        
        return self.rtfpathbuild(subfoldertlist, filename, filetype)
        
class FileStorageHandle():
    @staticmethod
    def saveto(data, path):
        with open(path, "xt") as f:
            pickle.dump(data, f)
    @staticmethod
    def savetobin(data, path):
        with open(path, "xb") as f:
            pickle.dump(data, f)
    @staticmethod
    def loadf(path):
        if os.path.exists(path):
            with open(path, "rt") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"{path} doesn't exist")
    @staticmethod
    def loadfbin(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"{path} doesn't exist")
        
    
        
class DataIO(Pathbuild):
    """_"IO tools between ram and file"_

    Methods:
        "tfread, tfwrite, dytfrw"
    """
    def __init__(self, dataonram, subpath_incl_filename_extension, isruntimepath=True, pathbystarttime=True):
        """_Init_

        Args:
            dataonram (_"any on ram"_): _"data on ram, runtime or define target"_
            subpath_incl_filename_extension (_"str or bool"_): _"relative folder location to file or True/False to autogenerate(True include runtime)"_
            radixChar (_str_): _"radix point selector, between . and ,"_
            isruntimepath (_bool_): _"do the names autogenerate by time value? if false, only strings are used to path.(default: True)"_
            pathbystarttime (_bool_): _"do the names autogenerate by start time value? if false, realtime is used to path.(default: True)"_
        """
        Pathbuild.__init__(self, isruntimepath)
        #temporary: process only str
        if not isinstance(subpath_incl_filename_extension, str):
            subpath_incl_filename_extension=str(subpath_incl_filename_extension)
        if os.curdir not in subpath_incl_filename_extension:
            subpath_incl_filename_extension=os.path.join(os.curdir, os.sep, subpath_incl_filename_extension)
        pathlist=subpath_incl_filename_extension.split(os.sep)
        folders=pathlist[0:-2] if len(pathlist)>2 else os.curdir
        fileN=pathlist[-1].split(os.extsep)[0]
        ftype=pathlist[-1].split(os.extsep)[-1]
        if not pathbystarttime:
            self.path=self.rtfpathbuild(folders, fileN, ftype)
        else:
            self.path=self.rtfpath_time(folders, fileN, ftype)
        self.ramdata=dataonram
        
        
    
    def tfread(self, binary=False):
        """_"file read process: multiprocessed easy version of torch.load"_
        
        Returns:
            _"tensor"_: _"runtime ram variable"_
        """
        path=self.path
        if not binary:
            return FileStorageHandle.loadf(path)
        elif binary:
            return FileStorageHandle.loadfbin(path)
        
    def tfwrite(self, binarymode=False):
        """_"file write process"_

        Returns:
            _file_: _"file in relative path, txt format"_
        """
        ramdata=self.ramdata
        path=self.path
        if not binarymode:
            FileStorageHandle.saveto(ramdata, path)
        elif binarymode:
            FileStorageHandle.savetobin(ramdata, path)