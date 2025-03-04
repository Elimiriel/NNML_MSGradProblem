import inspect;
import sys;
from . import tensormultiprocessings #import ResourceManager, Cacheing, istensorable, find_fastest_storage, filespaceckeck, getsignature, sysRateToNums, swapstorage, ramthreashold, vramthreshold, cpuworkers, gpuworkers;
from . import perfparameters #as pp;
from . import phyparameters #as phyp;
from . import timegen #import Curtimeout, Tpass, Starttime;
from . import dataiod #import DataIO, TensDFrame;
from . import model #import Model_Fn, DNNP;
from . import refdata #import Reference;
from . import teaching #import Train;


"""adding all in auto"""
__all__=[name for name, obj in inspect.getmembers(sys.modules[__name__]) if not name.startswith("_")];