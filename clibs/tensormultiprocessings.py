import os
#import sys
import gc
#import time
from datetime import datetime
import json
import perfparameters
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import polars as pl
#import io
#for searching fastest storage when OOM situations
import psutil
import tempfile
import uuid
import logging
from inspect import signature, isfunction, ismethod
import redis
import signal
from pathlib import Path
from multiprocessing import Pool, cpu_count
#do not call multiprocessing directly in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite


cpuworkers=mp.cpu_count()
gpuworkers=torch.cuda.device_count() if torch.cuda.is_available() else -1
#OS memory allocation and multiprocessing method setting
if os.name=="nt":
    mp.set_start_method(method="spawn", force=True)
else:
    mp.set_start_method(method="fork", force=True)


def istensorable(obj) -> bool:
    try:
        torch.tensor(obj)
        return True
    except (TypeError, ValueError):
        return False


ramthreashold=psutil.virtual_memory().total*perfparameters.cpu_reserve_fraction
vramthreshold=[torch.cuda.max_memory_allocated(ind) for ind in range(torch.cuda.device_count())] if torch.cuda.is_available() else [0]


def release_ram():
    """release cpu ram and return memory usage values.

    Returns:
        int: released cpu ram amounts in bytes.
    """
    used=psutil.virtual_memory().used
    gc.collect()
    cleaned=psutil.virtual_memory().used
    return used-cleaned

def release_vram(deviceid=None):
    """release vram and return memory usage values. can be used for multiple cuda devices.

    Args:
        deviceid (positive int, optional): target cuda device id. Defaults to None, then all devices.

    Returns:
        list: released vram amounts in bytes. index is cuda device number.
    """
    if not torch.cuda.is_available():
        return [0]
    cudanums = torch.cuda.device_count() if deviceid is None else deviceid
    used_memory=[]
    for ind in range(cudanums+1):
        used_memory.append(torch.cuda.memory_allocated(ind))
        torch.cuda.empty_cache()
        used_memory[ind]=used_memory[ind]-torch.cuda.memory_allocated(ind)
    return used_memory


class FileManager():
    """filepath manager / Storage management prototype during multiprocessing.
    """
    def __init__(self):
        self.files = set()

    def add_file(self, path: str|os.PathLike|Path):
        self.files.add(path)

    def remove_file(self, path: str|os.PathLike|Path):
        if os.path.exists(path):
            os.remove(path)
        self.files.discard(path)
        
    def get_state(self):
        return {"files": list(self.files)}

    def set_state(self, state):
        self.files = set(state.get("files", []))

    def _interupthandler(self):
        """detector of interrupts of main process.
        """
        signals=[signal.SIGINT, signal.SIGTERM]
        if os.name=="nt":
            signals.append(signal.SIGBREAK)
        for sig in signals:
            signal.signal(sig, self._exithandle)
        
    def _exithandle(self, signum, frame):
        """resource release handler as closer

        Args:
            signum (signal): _description_
            frame (_type_): _description_
        """
        self.clear_all()
        os._exit(1)

    def clear_all(self):
        """whole path cleaner."""
        with Pool(min(cpuworkers, cpu_count())) as pool:
            pool.map(os.remove, self.files)
        self.files.clear()
        gc.collect()
        
        
class ResourceManager(FileManager):
    def __init__(self):
        FileManager.__init__(self)

    def release_ram(self):
        """release cpu ram and return memory usage values.

        Returns:
            int: released cpu ram amounts in bytes.
        """
        return release_ram()

    def release_vram(self, deviceid=None):
        """release vram and return memory usage values. can be used for multiple cuda devices.


        Returns:
            list: released vram amounts in bytes. index is cuda device number.
        """
        return release_vram(deviceid)
    
    def ignore_signals(self):
        """Ignore termination signals to worker processes: only main process is controlled by interrupt handler based on signal module"""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        
    def clear_files(self):
        self.clear_all()
        

class Cacheing():
    def __init__(self, redihost="localhost", rediport=6379):
        self.redis_client=redis.StrictRedis(host=redihost, port=rediport, decode_responses=True)
        
    def _add(self, task_signature, memory_usage):
        """Add memory usage data for a task."""
        key=f"memory_cache:{task_signature}"
        self.redis_client.rpush(key, memory_usage)
        self.redis_client.expire(key, 3600)
    
    def _get_estimate(self, task_signature):
        """Get average memory usage for a task."""
        key=f"memory_cache:{task_signature}"
        if values := self.redis_client.execute_command("lrange", key, 0, -1):
            memvalues = [int(v) for v in values]
            return sum(memvalues) // len(memvalues)
        return None
        
    def _estimate_caching(self, task_signature, ram_used, vram_used):
        cached_memory = self._get_estimate(task_signature)
        if cached_memory is not None:
            return cached_memory  # Use cached memory estimate
        total_memory = ram_used + vram_used
        self._add(task_signature, total_memory)
        return total_memory


def check_available_storage():
    """ram and vram usage checking"""
    ramspace=psutil.virtual_memory().available
    vramspace=[torch.cuda.memory_allocated(ind) for ind in range(torch.cuda.device_count())] if torch.cuda.is_available() else [0]
    return ramspace, vramspace


def find_fastest_storage(forceswapsw=False, testsize=1E6):
    if not forceswapsw:
        if check_available_storage()[0]<=ramthreashold:
            """cpu is most preffered due to no limitations of functions"""
            return "cpu"
        if torch.cuda.is_available():
            """vram is 2nd preffered due to limitations of functions. most empty vram is selected"""
            maxid=check_available_storage()[1].index(max(check_available_storage()[1]))
            return f"cuda:{maxid}"
    else:
        """case of storage use because of ram shortage"""
        def find_drives():
            #drive search
            partitions=psutil.disk_partitions()
            drivers=[p.mountpoint for p in partitions if os.access(p.mountpoint, os.W_OK)]
            return drivers

        canditates=find_drives()+["/dev/shm", tempfile.gettempdir(), "/tmp"]
        canditates=[path for path in canditates if os.path.exists(path) and os.access(path, os.W_OK)]
        testdata=b"x"*int(testsize)
        times=[]
        for path in canditates:
            try:
                filepath=os.path.join(path, "test.tmp")
                stime=datetime.now().timestamp()
                torch.save(testdata, filepath)
                torch.load(filepath, weights_only=True)
                os.remove(filepath)
                times.append(datetime.now().timestamp()-stime)
            except Exception:
                times.append(datetime.max.timestamp())
        return canditates[times.index(min(times))] if times else tempfile.gettempdir()
        
        
def generate_disk_path(base_path: str|os.PathLike|Path, *file_name):
    return Path(base_path).joinpath(f"{file_name}.pt")


def setup_log(log_file):
    logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

        
def worker_elog(taskid, e):
    logger=logging.getLogger()
    logger.error(f"{taskid}_RAM usage: {check_available_storage()[0] / (1024 * 1024):.2f} MB\n")
    for ind in range(torch.cuda.device_count()+1):
        logger.error(f"{taskid}_VRAM usage: {check_available_storage()[1][ind] / (1024 * 1024):.2f} MB\n")
    logger.error(f"Worker {taskid} error: {e}")


swapstorage=find_fastest_storage(forceswapsw=True)

    
def filespaceckeck(path=swapstorage):
    try:
        freespace=psutil.disk_usage(path)
        return freespace>=int(psutil.disk_usage(path).total)
    except Exception as e:
        logging.error(f"ERR on ckecking space of {path}: {e}")
        return 0
    
    
def _get_optimal_batch_size(total_tasks, max_memory_per_batch, task_memory_estimate):
    """Calculate the optimal batch size based on memory constraints."""
    max_tasks_per_batch = max_memory_per_batch // task_memory_estimate
    divisors = [d for d in range(1, total_tasks + 1) if total_tasks % d == 0]
    return max([d for d in divisors if d <= max_tasks_per_batch], default=1)
    
        
def getsignature(target):
    if isfunction(target) or ismethod(target):
        return signature(target)
    elif hasattr(target, "__call__"):
        """callable objects"""
        return signature(target.__call__)
    else:
        raise ValueError(f"can't extract sign from {target}")
    
    
def sysRateToNums(cworkrate: int=0.3, gworkrate: int=0.3):
    """performance adjuster of multitasking

    Args:
        cworkrate (_float_): rate of cpu usage in 0.0-1.0.
        gworkrate (_float_): rate of gpu usage in 0.0-1.0.

    Raises:
        ValueError: out of rate range.

    Returns:
        _|(int, int)_: cpu and gpu usage nums in total resources.
    """
    if not (0.0 <= cworkrate <= 1.0) or not (0.0 <= gworkrate <= 1.0):
        raise ValueError(f"input desired usage rate of the system in 0.0-1.0. ex)0.5 means 50% of full resource. input: {cworkrate}, {gworkrate}")
    cworkers=int(cpuworkers*cworkrate)
    gworkers=int(gpuworkers*gworkrate)
    return cworkers, gworkers


class ProtoMP(Cacheing, ResourceManager):
    def __init__(self, Clsfunc, cworkers=None, gworkers=None, redihost="localhost", rediport=6379, *args, **kwargs):
        Cacheing.__init__(self, redihost, rediport)
        ResourceManager.__init__(self)
        self.cworkers=cworkers or cpuworkers
        self.gworkers=gworkers or gpuworkers
        self.Clsfunc=Clsfunc
        self.shared_args=self._shareallocate(args)
        self.shared_kwargs=self._shareallocate(kwargs)
    
    @abs
    def _shareallocate(self, obj):
        pass
    
    @abs
    def multiprocess(self):
        pass
    
    @abs
    def _shared_allocate(self, obj):
        """manual shared memory allocation even in Windows: in plan with Rust or C++"""
        pass
    @abs
    def multiprocess_wrapper(self, 
                             Clsfunc: nn.Module=None, 
                             batch=0, *args, **kwargs):
        """Multiprocessing wrappwer even in Windows: in plan with Rust or C++ or other libs"""
    
    def __del__(self):
        try:
            self.redis_client.close()
            self.clear_files()
        except Exception as e:
            logging.error(f"ERR on cleanup {e}")