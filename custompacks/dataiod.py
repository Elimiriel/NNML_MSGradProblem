import os;
from .tensormultiprocessings import RayTorchMP, cpuworkers, ResourceManager, raydeclare;
import torch;
import seaborn as sns;
import pandas as pd;
import traceback;
import pickle;
from .timegen import Curtimeout, Starttime;
from torch.utils.data import Dataset, DataLoader;
import numpy as np;
import ray;
from matplotlib import pyplot as plt, font_manager as fm;
from varname import nameof;

class Pathbuild():
    def __init__(self, pathbyruntime):
        """_"relative pathbuilder can include times"_

        Args:
            pathbyruntime (_Bool_): _"True: use"_
        """
        curtimepath=Curtimeout().str;
        if pathbyruntime:
            self.curtimepath=curtimepath;
        else:
            self.curtimepath=Starttime.str;
        
    def stOrtfpathbuild(self, subfolderlist, filename, filetype):
        """do not call this method directly out of the class!"""
        #subfolds=[];
        #filenames=[];
        #filetypes=[];
        #for ind, var in enumerate(subfolderlist):
        subfolderlist=f"{subfolderlist}+{os.sep}";
        #for ind, var in enumerate(subfolderlist):
            #filenames[ind]=filename[ind];
        #for ind, var in enumerate(filetype):
            #filetypes[ind]=os.extsep+filetype[ind];
        res=f"{os.curdir}+{os.sep}+{subfolderlist}+{filename}+{os.extsep}+{filetype}";
        #res=[os.curdir+os.sep+f"{subfolds[ind]}"+f"{filenames[ind2]}"+os.extsep+f"{filetypes[ind3]}" for ind, ind2, ind3 in range(0, len(subfolds), 1),  range(0, len(filenames), 1)),  range(0, len(filetypes, 1)];
        return res;
    
    def __listORstr(self, pathlist, additionals=""):
        if not isinstance(pathlist, str):
            res=[f"{additionals}"+f"{pathlist[ind]}" for ind in range(0, len(pathlist), 1) ];
        else:
            res=f"{additionals+pathlist}"
        return res;
    
    def rtfpathbuild(self, subfolderlist, filename, filetype):
        """_"multifile/folder pathbuilder"_

        Args:
            subfolderlist (_"list of str"_): _targetsubfolders_
            filename (_"list of str"_): _targetfilenames_
            filetype (_"list of str"_): _targetextensions_

        Returns:
            _"list of str"_: _"path list"_
        """
        subfolderlist=self.__listORstr(subfolderlist);
        filename=self.__listORstr(filename);
        filetype=self.__listORstr(filetype);
        
        return self.stOrtfpathbuild(subfolderlist, filename, filetype);
        
    def rtfpath_time(self, subfolderlist, filename, filetype):
        """_"runtime included multifile/folder pathbuilder"_

        Args:
            subfolderlist (_"list of str"_): _targetsubfolders_
            filename (_"list of str"_): _targetfilenames_
            filetype (_"list of str"_): _targetextensions_

        Returns:
            _"list of str"_: _"path list, runtime included"_
        """
        subfoldertlist=self.__listORstr(subfolderlist, self.curtimepath);
        filename=self.__listORstr(filename, self.curtimepath);
        filetype=self.__listORstr(filetype);
        
        return self.rtfpathbuild(subfoldertlist, filename, filetype);
        
class FileStorageHandle():
    @staticmethod
    def saveto(data, path):
        with open(path, "xt") as f:
            pickle(data, f);
    @staticmethod
    def savetobin(data, path):
        with open(path, "xb") as f:
            pickle(data, f);
    @staticmethod
    def loadf(path):
        if os.path.exists(path):
            with open(path, "rt") as f:
                return pickle.load(f);
        else:
            raise FileNotFoundError(f"{path} doesn't exist");
    @staticmethod
    def loadfbin(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f);
        else:
            raise FileNotFoundError(f"{path} doesn't exist");
        

class ParaPrepset(Dataset, ResourceManager):
    def __init__(self, storepath, format, cworkR=0.3, gworkR=0.3, preprocess=None, *data):
        """multiprocessed Dataset. do not call multiprocessing directly in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite.

        Args:
            storepath (_storagepath_, optional): _path to save_. Defaults to swapstorage.
            format (_str_): save format type. b for binary, t for text.
            xworkR (_float_): rate of system use. c is cpu, g is gpu.
            preprocess (_class|method|func_, optional): _preprocessing tasks_. Defaults to None.
            data (_any_): targetdata to save.
        """
        Dataset.__init__(self);
        ResourceManager.__init__(self);
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True);
        self.data = [data];
        self.obj=preprocess;
        self.storepath=storepath;
        os.makedirs(self.storepath, exist_ok=True);
        self._preprocess=raydeclare(self._protopreprocess, cworkR, gworkR)(len(data), data);
        if not isinstance(format, ["t", "b"]):
            raise ValueError("format must be t for text or b for binary");
        self.format=f"{format}"
        
    @ray.remote
    def _protopreprocess(self, tid, item):
        # 데이터를 전처리하는 함수
        """for worker-interrupt relations"""
        self.ignore_signals();
        filename=os.path.join(self.storepath, f"{tid}.pt");
        tempfilename=os.path.join(filename, ".tmp");
        if self.obj:
            processed=self.obj(item);
        else:
            processed=item;
        if not os.path.exists(filename):
            """preventing file crash: save actual name after the save is ok"""
            torch.save(processed, tempfilename);
            os.rename(tempfilename, filename);
        else:
            torch.save(processed, filename);
        self.file_manager.add_file(filename);
        return filename;
    
    def __len__(self):
        return len(self.data);
    
    def __getitem__(self, idx):
        # 데이터 병렬 전처리 수행
        tasks=[self._preprocess.remote(i, item) for i, item in enumerate(self.data)];
        files=ray.get(tasks);
        target_file=files[idx];
        if os.path.exists(target_file):
            with open(target_file, "r".join(self.format)) as f:
                res=torch.load(f, weights_only=True, map_location="cpu");
            return res;
        else:
            raise FileNotFoundError(f"Process file {target_file} not found");
    def __del__(self):
            self.clear_files();
            if ray.is_initialized():
                ray.shutdown();
    
class ParaPrepLoad(DataLoader):
    def __init__(self, data, loadlocation="cpu", batches=64, workerR=0.5):
        """multiprocessed Dataloader

        Args:
            data (Dataset|ParePreset): _reading target_
            loadlocation (str, optional): _cpu|cuda_. Defaults to "cpu".
            batches (int, optional): _nums of batches_. Defaults to 64.
            workerR (float, optional): _rate of cpu usage_. Defaults to 0.5.
        """
        pindevice=torch.device(loadlocation);
        pinmemory= loadlocation!="cpu";
        workers, _=raydeclare(None, workerR, workerR);
        DataLoader.__init__(self, data, batch=batches, num_workers=workers, pin_memory=pinmemory, pin_memory_device=pindevice);
        
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
        Pathbuild.__init__(self, isruntimepath);
        #temporary: process only str
        if not isinstance(subpath_incl_filename_extension, str):
            subpath_incl_filename_extension=str(subpath_incl_filename_extension);
        if os.curdir not in subpath_incl_filename_extension:
            subpath_incl_filename_extension=os.path.join(os.curdir, os.sep, subpath_incl_filename_extension);
        pathlist=subpath_incl_filename_extension.split(os.sep);
        folders=pathlist[0:-2] if len(pathlist)>2 else os.curdir;
        fileN=pathlist[-1].split(os.extsep)[0];
        ftype=pathlist[-1].split(os.extsep)[-1];
        if not pathbystarttime:
            self.path=self.rtfpathbuild(folders, fileN, ftype);
        else:
            self.path=self.rtfpath_time(folders, fileN, ftype);
        self.ramdata=dataonram;
        
        
    
    def tfread(self, binary=False):
        """_"file read process: multiprocessed easy version of torch.load"_
        
        Returns:
            _"tensor"_: _"runtime ram variable"_
        """
        path=self.path;
        if not binary:
            return FileStorageHandle.loadf(path);
        elif binary:
            return FileStorageHandle.loadfbin(path);
        
    def tfwrite(self, binarymode=False):
        """_"file write process"_

        Returns:
            _file_: _"file in relative path, txt format"_
        """
        ramdata=self.ramdata;
        path=self.path;
        if not binarymode:
            FileStorageHandle.saveto(ramdata, path);
        elif binarymode:
            FileStorageHandle.savetobin(ramdata, path);

class TensDFrame(Pathbuild):
    """tensordata to database and draw seaborn graph

    Raises:
        TypeError: _"type mismatch to sns.relplot"_

    Returns:
        _FacetGrid_: _"seaborn graph"_
    """
    
    def __init__(self, pathbystart, save, *tensordata, cpuR=0.3, gpuR=0.3):
        """_"tensor to pd.DataFrame, and seaborn to plot"_

        Args:
            pathbystart (_bool_): _"name by starttime if true, runtime if false"_
            save (_bool_): _"runtime switch after initialization, to control saving graph"_

        Raises:
            TypeError: _"wrong type for drawseaborn method"_

        Returns:
            _FacetGrid_: _"seaborn graph from drawseaborn method"_
        """
    #바로 *변수들을 받으면, 이들은 튜플로 묶인 상태
    #클래스의 로컬작업은 병렬결과를 모으는 단계가 막혀 메소드나 외부로 빼야 함
        #alldata=torch.stack(tensordata, dim=-1);
        Pathbuild.__init__(self, pathbystart);
        usingGffam=[f.name for f in fm.fontManager.ttflist if "Pretendard" or "Segoe" in f.name] or None;
        if usingGffam and usingGffam.index("Pretendard JP Variable"):
            plt.rcParams["font.family"]="Pretendard JP Variable";
        else:
            plt.rcParams["font.family"]="Segoe UI Variable Text";
        plt.rc('font', size=15)        # 기본 폰트 크기
        plt.rc('axes', labelsize=16)   # x,y축 label 폰트 크기
        plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기 
        plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
        plt.rc('legend', fontsize=14)  # 범례 폰트 크기
        plt.rc('figure', titlesize=17) # figure title 폰트 크기
        inputs=torch.stack(tensordata[:])
        inputvarnames=[nameof(tensordata[:])]
        self.data=inputs;
        self.inputvarnames=inputvarnames
        self.save=save;
    
    def drawseaborn(self, kind, xlabel, ylabel, hue, legend=None, hueorder=None, size=None, sizeorder=None, style=None, styleorder=None, units=None, weights=None, row=None, col=None, rorder=None, corder=None, palette=None, marker=None, dash=None, height=None, aspect=None, facet_kws=None):
        """_Graph_

        Args:
            legend (_str_): _description_
            kind (_str_): _"scatter or line"_
            xlabel (_str_): _description_
            ylabel (_str_): _description_
            hue (_str_): _"kinds of ydata"_
            hueorder (_iters_): _"order of hue. itered hue values"_
            size (_str_): _description_
            sizeorder (_iters_): _description_
            style (_str_): _description_
            styleorder (_iters_): _description_
            units (_str_): _description_
            weights (_str_): _description_
            row (_str_): _"(maybe)label of multiple graphs in col locations, different x"_
            col (_str_): _"label of multiple graphs in col locations, different y"_
            rorder (_iters_): _"order of row. itered row vals"_
            corder (_iters_): _"order of cols, itered col vals"_
            palette (_Colormap_): _description_
            marker (_markers_): _description_
            dash (_dashes_): _description_
            height (_float_): _description_
            aspect (_aspect_): _description_
            facet_kws (_dict_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
        """
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        #if isinstance([legend, xlabel, ylabel, row, col, kind, units, hue, size, weights], str) | isinstance([aspect, height], float):
        #    raise TypeError(traceback.format_exc());
        data=torch.stack(self.data[:]).numpy();
        data=pd.DataFrame(data.T, columns=[xlabel, self.inputvarnames[:]]);#무조건 마지막 차원을 열로 인식하므로, 행으로 인식하도록 Transpose
        long_data = data.melt(id_vars=xlabel, var_name=hue, value_name=ylabel);
        sns.set_theme({"paper":"paper"}, {"style": "whitegrid"}, "deep", "Pretendart JP Variable", 1.0, True);
        Plot=sns.relplot(long_data, kind=kind, x=xlabel, y=ylabel, hue=hue, hue_order=hueorder, size=size, size_order=sizeorder, style=style, style_order=styleorder, units=units, height=height, weights=weights, row=row, col=col, row_order=rorder, col_order=corder, palette=palette, markers=marker, dashes=dash, aspect=aspect, facet_kws=facet_kws);
        Plot;
        #if self.save:
        #    Plot.savefig(self.rtfpath_time("Graph", f"output_{hue}-{legend}, {ylabel}", f"{}png"));
        os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"