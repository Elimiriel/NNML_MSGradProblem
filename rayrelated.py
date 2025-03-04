import os
#import sys;
import gc;
#import time;
from datetime import datetime;
import json;
import torch
import torch.nn as nn
import torch.multiprocessing as mp;
import ray;
#from ray import tune;
#import io;
#for searching fastest storage when OOM situations
import psutil;
import tempfile;
from varname import nameof;
import uuid;
import logging;
from inspect import signature, isfunction, ismethod;
import redis;
import signal;

def raydeclare(targetfunction, cworkrate=0.3, gworkrate=0.3):
    """ray declare wrapper or performance adjuster of already declared ray

    Args:
        targetfunction (_func|method|class_): target to being worker. None to performance parameter return mode.
        cworkrate (_float_): rate of cpu usage in 0.0-1.0.
        gworkrate (_float_): rate of gpu usage in 0.0-1.0.

    Raises:
        ValueError: out of rate range.

    Returns:
        _rayworker|(int, int)_: ray-wrapped worker in default, cpu and gpu usage rate in parameter return mode.
    """
    if not (0.0 <= cworkrate <= 1.0) or not (0.0 <= gworkrate <= 1.0):
        raise ValueError(f"input desired usage rate of the system in 0.0-1.0. ex)0.5 means 50% of full resource. input: {cworkrate}, {gworkrate}")
    cworkers=int(cpuworkers*cworkrate);
    gworkers=int(gpuworkers*gworkrate);
    if targetfunction is not None:
        return ray.remote(num_cpus=cworkers, num_gpus=gworkers)(targetfunction);
    else:
        return cworkers, gworkers;

@ray.remote
class Rayworker():
    def __init__(self, RManager):
        """Ray Actor resource supporter. collectedres is merged results.


        Args:
            RManager (_type_): _description_

        Raises:
            TypeError: _description_
        """
        if isinstance(RManager, ResourceManager):
            self.ResourceManager=RManager;
        else:
            raise TypeError("Must have ResourceManager");
        self._worker=[];
        self._workerstatus={};
        self._savefilepath=[];
        self._runid=str(uuid.uuid4()).join(str(uuid.uuid5(uuid.NAMESPACE_DNS, "Actor")));
        
    def _idleworker(self, Actor):
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        """dict to list"""
        wid, status = zip(*self._workerstatus.items());
        wid=torch.tensor(wid, dtype=torch.int64);
        status=torch.tensor([st.encode("utf-8") for st in status], dtype=torch.uint8);
        idle_indices = (status == torch.tensor("idle".encode("utf-8"))).nonzero(as_tuple=True)[0]
        if len(idle_indices)==0:
            wid=self._newwork(Actor);
        else:
            wid=wid[idle_indices[0]].item();
        return wid;
    
    def _marktask(self, taskid, status):
        self._workerstatus[taskid]=str(status);
    
    def _newwork(self, Actor):
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        worker=Actor.remote();
        """worker and path register"""
        self._worker.append(worker);
        self._savefilepath.append(None);
        workerid=len(self._worker)-1;
        self._marktask(workerid, "idle");
        return workerid;
    
    def _reshandle(self, eres, device, taskid):
        def unithand():
            """taskres place manager"""
            result=ray.get(eres);
            """typeconvert to device management"""
            if not istensorable(result):
                result=torch.tensor(result.encode("utf-8"), dtype=torch.unit8);
            elif not isinstance(result, torch.Tensor):
                result=torch.tensor(result);
                
            """storage device management"""
            if device!=swapstorage:
                result=result.to(device);
            else:
                """filebase tasking"""
                filepath=self._saveres(result, swapstorage, taskid);
                result=filepath;
            return result;
        return ray.remote(unithand).remote();
        
    def _assign_work(self, Actor, args, kwargs, submethod=None):
        """in actual work, external ray Actor is actual worker"""
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        device=find_fastest_storage();
        taskid=self._idleworker(Actor);
        worker=self._worker[taskid];
        self._marktask(taskid, "busy");
        try:
            if submethod is None:
                ray_taskres=worker.remote(*args, **kwargs);
            elif hasattr(Actor, submethod):
                ray_taskres=getattr(worker, submethod).submethod.remote(*args, **kwargs);
            else:
                """when Actor don't have disignated submethod"""
                self._marktask(taskid, "idle");
                raise AttributeError("invalid method");
            """storing references"""
            self._workerstatus[taskid]=ray_taskres;
            
            """handling process apply"""
            ray.wait([ray_taskres]);
            processed=self._reshandle(ray_taskres, device, taskid);
            
            self._marktask(taskid, "idle");
            return taskid, ray_taskres, processed;
        except Exception as e:
            worker_elog(taskid, e);
            print(f"Worker {taskid} error: {e}");
            self._marktask(taskid, "idle");
            self._workerstatus[taskid]=None;
            return taskid, e;
    
    def _saveres(self, res, diskpath, tid):
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        savepath=os.path.join(diskpath, f"{self._runid}-{self.taskname}-{tid}.pt");
        torch.save(res.cpu(), savepath);
        self.savefilepath[tid]=savepath;
        self.ResourceManager.file_manager.add_file(savepath);
        return savepath;
        
    def _fileload(self, taskid):
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        """Load results from a file."""
        filepath = self._savefilepath[taskid] if taskid < len(self._savefilepath) else None

        if filepath and os.path.exists(filepath):
            try:
                return torch.load(filepath, map_location="cpu", weights_only=True);
            except Exception as e:
                logging.error(f"Error loading worker results for task {taskid}: {e}")
                return e
        return None
        
    def _merge_results(self, refs, fileres):
        """Merge in-memory and file-based results."""
        return [refs[i] if refs[i] is not None else fileres[i] for i in range(max(len(refs), len(fileres)))];
    
        
    def collectedres(self, Actor, submethod=None, *args, **kwargs):
        """interrupt handler handling: subprocess ignores and controlled by the main"""
        self.ResourceManager.ignore_signals();
        # Assign work to an actor and get references
        taskid, raw_ref, processed_ref = self._assign_work(Actor, args, kwargs, submethod)

        # Retrieve processed results
        try:
            processed_result = ray.get(processed_ref)
        except Exception as e:
            print(f"Error retrieving processed result for task {taskid}: {e}")
            processed_result = None

        # Load results from file for completed tasks
        try:
            file_result = ray.get(self._fileload(taskid))
        except Exception as e:
            print(f"Error loading file result for task {taskid}: {e}")
            file_result = None
        
        """sync"""
        ray.wait(processed_result, file_result, num_returns=len(processed_result)+len(file_result));
        torch.cuda.synchronize();

        # Merge in-memory and file-based results within Rayworker
        return self._merge_results([processed_result], [file_result])

@ray.remote
class WorkerresConv():
    def __init__(self, ResourceManager):
        self.ResourceManager=ResourceManager;
        
    def cprocess(self, taskid, result):
        self.ResourceManager.ignore_signals();
        try:
            if isinstance(result, torch.uint8):
                return bytes(result.tolist());
            if isinstance(result, Exception):
                logging.error(f"ERR in worker result {taskid}: {result}")
                logging.error(f"ERR restore str {taskid}: {e}");
                return None;
        except Exception as e:
            logging.error(f"processing error {taskid}: {e}")
            return None;
@ray.remote
class FMwrap():
    def __init__(self, state=None):
        self.files = set(state.get("files", [])) if state else set()

    def add_file(self, path):
        self.files.add(path)

    def remove_file(self, path):
        if path in self.files:
            self.files.remove(path)

    def get_state(self):
        return {"files": list(self.files)}

@ray.remote
class ResourceManagerWrapper():
    """Wrapper to transfer only ResourceManager of the main class"""
    def __init__(self, state=None):
        self.file_manager = FMwrap.remote(state.get("file_manager", {})) if state else FMwrap.remote()

    def file_manager_add(self, path):
        ray.get(self.file_manager.add_file.remote(path))

    def get_state(self):
        return ray.get(self.file_manager.get_state.remote())
    
    def update_state(self, state):
        if "file_manager" in state:
            ray.get(self.file_manager.set_state.remote(state["file_manager"]))