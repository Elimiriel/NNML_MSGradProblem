import os;
#import sys;
import gc;
#import time;
from datetime import datetime;
import json;
import perfparameters;
import torch;
import torch.nn as nn;
import torch.multiprocessing as mp;
import ray;
#from ray import tune;
#import io;
#for searching fastest storage when OOM situations
import psutil;
import tempfile;
from varname.core import nameof;
import uuid;
import logging;
from inspect import signature, isfunction, ismethod;
import redis;
import signal;
import abc;
#do not call multiprocessing directly in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite
cpuworkers=mp.cpu_count();
gpuworkers=torch.cuda.device_count() if torch.cuda.is_available() else -1;
#OS memory allocation and multiprocessing method setting
if os.name=="nt":
    mp.set_start_method(method="spawn", force=True);
else:
    mp.set_start_method(method="fork", force=True);

def istensorable(obj):
    try:
        torch.tensor(obj);
        return True;
    except (TypeError, ValueError):
        return False;

ramthreashold=psutil.virtual_memory().total*perfparameters.cpu_reserve_fraction;
vramthreshold=[torch.cuda.max_memory_allocated(ind) for ind in range(torch.cuda.device_count())] if torch.cuda.is_available() else [0];

def release_ram():
    """release cpu ram and return memory usage values.

    Returns:
        int: released cpu ram amounts in bytes.
    """
    used=psutil.virtual_memory().used;
    gc.collect();
    cleaned=psutil.virtual_memory().used;
    return used-cleaned;

def release_vram(deviceid=None):
    """release vram and return memory usage values. can be used for multiple cuda devices.

    Args:
        deviceid (positive int, optional): target cuda device id. Defaults to None, then all devices.

    Returns:
        list: released vram amounts in bytes. index is cuda device number.
    """
    if not torch.cuda.is_available():
        return [0];
    cudanums = torch.cuda.device_count() if deviceid is None else deviceid;
    used_memory=[];
    for ind in range(cudanums+1):
        used_memory.append(torch.cuda.memory_allocated(ind));
        torch.cuda.empty_cache();
        used_memory[ind]=used_memory[ind]-torch.cuda.memory_allocated(ind);
    return used_memory;

class FileManager():
    """filepath manager / Storage management prototype during multiprocessing.
    """
    def __init__(self):
        self.files = set()

    def add_file(self, path):
        self.files.add(path)

    def remove_file(self, path):
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
        signals=[signal.SIGINT, signal.SIGTERM];
        if os.name=="nt":
            signals.append(signal.SIGBREAK);
        for sig in signals:
            signal.signal(sig, self._exithandle);
        
    def _exithandle(self, signum, frame):
        """resource release handler as closer, including ray shutdown

        Args:
            signum (signal): _description_
            frame (_type_): _description_
        """
        self.clear_all();
        if ray.is_initialized():
            ray.shutdown();
        os._exit(1);

    def clear_all(self):
        """whole path cleaner."""
        for path in list(self.files):
            if os.path.exists(path):
                os.remove(path)
            self.files.discard(path)
        
class ResourceManager(FileManager):
    def __init__(self):
        FileManager.__init__(self);

    def release_ram(self):
        """release cpu ram and return memory usage values.

        Returns:
            int: released cpu ram amounts in bytes.
        """
        return release_ram();

    def release_vram(self, deviceid=None):
        """release vram and return memory usage values. can be used for multiple cuda devices.


        Returns:
            list: released vram amounts in bytes. index is cuda device number.
        """
        return release_vram(deviceid);
    
    def ignore_signals(self):
        """Ignore termination signals to worker processes: only main process is controlled by interrupt handler based on signal module"""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        
    def clear_files(self):
        self.clear_all();

class Cacheing():
    def __init__(self, redihost="localhost", rediport=6379):
        self.redis_client=redis.StrictRedis(host=redihost, port=rediport, decode_responses=True);
        
    def _add(self, task_signature, memory_usage):
        """Add memory usage data for a task."""
        key=f"memory_cache:{task_signature}"
        self.redis_client.rpush(key, memory_usage);
        self.redis_client.expire(key, 3600);
    
    def _get_estimate(self, task_signature):
        """Get average memory usage for a task."""
        key=f"memory_cache:{task_signature}"
        if values := self.redis_client.execute_command("lrange", key, 0, -1):
            memvalues = [int(v) for v in values];
            return sum(memvalues) // len(memvalues);
        return None;
        
    def _estimate_caching(self, task_signature, ram_used, vram_used):
        cached_memory = self._get_estimate(task_signature)
        if cached_memory is not None:
            return cached_memory  # Use cached memory estimate
        total_memory = ram_used + vram_used
        self._add(task_signature, total_memory)
        return total_memory

def check_available_storage():
    """ram and vram usage checking"""
    ramspace=psutil.virtual_memory().available;
    vramspace=[torch.cuda.memory_allocated(ind) for ind in range(torch.cuda.device_count())] if torch.cuda.is_available() else [0];
    return ramspace, vramspace;

def find_fastest_storage(forceswapsw=False, testsize=1E6):
    if not forceswapsw:
        if check_available_storage()[0]<=ramthreashold:
            """cpu is most preffered due to no limitations of functions"""
            return "cpu"
        if torch.cuda.is_available():
            """vram is 2nd preffered due to limitations of functions. most empty vram is selected"""
            maxid=check_available_storage()[1].index(max(check_available_storage()[1]));
            return f"cuda:{maxid}"
    else:
        """case of storage use because of ram shortage"""
        def find_drives():
            #drive search
            partitions=psutil.disk_partitions();
            drivers=[p.mountpoint for p in partitions if os.access(p.mountpoint, os.W_OK)];
            return drivers;

        canditates=find_drives()+["/dev/shm", tempfile.gettempdir(), "/tmp"];
        canditates=[path for path in canditates if os.path.exists(path) and os.access(path, os.W_OK)]
        testdata=b"x"*int(testsize);
        times=[];
        for path in canditates:
            try:
                filepath=os.path.join(path, "test.tmp");
                stime=datetime.now().timestamp();
                torch.save(testdata, filepath);
                torch.load(filepath, weights_only=True);
                os.remove(filepath);
                times.append(datetime.now().timestamp()-stime);
            except Exception:
                times.append(datetime.max.timestamp());
        return canditates[times.index(min(times))] if times else tempfile.gettempdir();
        
def generate_disk_path(base_path, *file_name):
    return os.path.join(base_path, f"{file_name}.pt");

def setup_log(log_file):
    logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s");
        
def worker_elog(taskid, e):
    logger=logging.getLogger();
    logger.error(f"{taskid}_RAM usage: {check_available_storage()[0] / (1024 * 1024):.2f} MB\n");
    for ind in range(torch.cuda.device_count()+1):
        logger.error(f"{taskid}_VRAM usage: {check_available_storage()[1][ind] / (1024 * 1024):.2f} MB\n");
    logger.error(f"Worker {taskid} error: {e}");

swapstorage=find_fastest_storage(forceswapsw=True);
    
def filespaceckeck(path=swapstorage):
    try:
        freespace=psutil.disk_usage(path);
        return freespace>=int(psutil.disk_usage(path).total);
    except Exception as e:
        logging.error(f"ERR on ckecking space of {path}: {e}");
        return 0;
    
def _get_optimal_batch_size(total_tasks, max_memory_per_batch, task_memory_estimate):
    """Calculate the optimal batch size based on memory constraints."""
    max_tasks_per_batch = max_memory_per_batch // task_memory_estimate
    divisors = [d for d in range(1, total_tasks + 1) if total_tasks % d == 0]
    optimal_batch_size = max([d for d in divisors if d <= max_tasks_per_batch], default=1)
    return optimal_batch_size
    
        
def getsignature(target):
    if isfunction(target) or ismethod(target):
        return signature(target);
    elif hasattr(target, "__call__"):
        """callable objects"""
        return signature(target.__call__);
    else:
        raise ValueError(f"can't extract sign from {target}");
    
def sysRateToNums(cworkrate=0.3, gworkrate=0.3):
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
    cworkers=int(cpuworkers*cworkrate);
    gworkers=int(gpuworkers*gworkrate);
    return cworkers, gworkers;

def raydeclare(targetfunction, cworkrate=0.3, gworkrate=0.3):
    """ray performance wrapper to function or class

    Args:
        targetfunction (_func|method|class_): target to being worker.
        cworkrate (_float_): rate of cpu usage in 0.0-1.0. under 0.1 means sharing same process with other works.
        gworkrate (_float_): rate of gpu usage in 0.0-1.0. under 0.1 means sharing same process with other works.

    Raises:
        ValueError: out of rate range.

    Returns:
        __Rayworker|(float, float)_: ray-wrapped worker in default, cpu and gpu usage rate in parameter return mode.
    """
    if not (0.0 <= cworkrate <= 1.0) or not (0.0 <= gworkrate <= 1.0):
        raise ValueError(f"input desired usage rate of the system in 0.0-1.0. ex)0.5 means 50% of full resource. input: {cworkrate}, {gworkrate}")
    cworkers=int(cpuworkers*cworkrate);
    gworkers=int(gpuworkers*gworkrate);
    return ray.remote(num_cpus=cworkers, num_gpus=gworkers)(targetfunction);


@ray.remote
class _Rayworker():
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
        #runid=str(uuid.uuid4()).join(str(uuid.uuid5(uuid.NAMESPACE_DNS, "results")));
        setup_log(os.path.join(tempfile.gettempdir(), f"{self.runid}.log"));
        
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
        self.ResourceManager.add_file(savepath);
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

        # Merge in-memory and file-based results within _Rayworker
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

class ProtoMP(Cacheing, ResourceManager):
    def __init__(self, Clsfunc, cworkers=None, gworkers=None, redihost="localhost", rediport=6379, *args, **kwargs):
        Cacheing.__init__(self, redihost, rediport);
        ResourceManager.__init__(self);
        self.cworkers=cworkers or cpuworkers;
        self.gworkers=gworkers or gpuworkers;
        self.Clsfunc=Clsfunc;
        self.args=args;
        self.kwargs=kwargs;
    
    @abc.abstractmethod
    def _shareallocate(self, obj):
        pass;
    
    @abc.abstractmethod
    def multiprocess(self):
        pass;
    
    def __del__(self):
        try:
            self.redis_client.close();
            self.clear_files();
        except Exception as e:
            logging.error(f"ERR on cleanup {e}");

class RayTorchMP(ProtoMP):
    def __init__(self, Clsfunc=None, raydefault=True, cworkers=None, gworkers=None, redihost="localhost", rediport=6379, *args, **kwargs):
        """_"Multiprocess function runner, GPT 한마디: 클래스의 메소드는 직렬화가 불가능하므로 이 클래스 상속 후 클래스 내부에서 사용해야 함, @staticmethod는 self와의 연결을 끊어 학습용으로 사용불가
        Due to limitations about methods in classes, unserializable from outsode, herit this class and use multiprocess method.
        do not call multiprocessing methods directly global in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite.
        and for only global funcs, filling initial func and args
        torch.save and torch.load were used-gpu tensor should be moved to cpu"_
        
        Args:
            process (_ray(func|class)_): target to being calculated. must be wrapped with @ray.remote
        """
        if raydefault and not ray.is_initialized():
            ray.init(_system_config={"object_spilling_config": json.dumps({"type": "filesystem", "params": {"directory_path": swapstorage}})}, ignore_reinit_error=True);
        ProtoMP.__init__(self, Clsfunc, cworkers, gworkers, redihost, rediport);
        if Clsfunc is not None:
            self.worker=_Rayworker.options(num_cpus=self.cworkers, num_gpus=self.gworkers).remote(self);
        
    def _shareallocate(self, obj):
        """moving obj to shared space"""
        device=find_fastest_storage();
        if device==swapstorage:
            storepath=os.path.join(swapstorage, os.sep, f"{nameof(obj)}", os.sep);
        
        if isinstance(obj, (torch.Tensor, nn.Module)):
            """already tensor or nn.Module class"""
            if not storepath:
                sharedobj=obj.share_memory_().to(device);
            else:
                filepath=generate_disk_path(self.diskpath, f".{nameof(obj)}.pt");
                tobj=obj.cpu();
                sharedobj=torch.save(tobj, filepath);
                self.file_manager.add_file(filepath);
            return sharedobj;
        elif istensorable(obj):
            """tensor convertable objs"""
            if not storepath:
                sharedobj=torch.tensor(obj).share_memory_().to(device);
            elif filespaceckeck():
                filepath=generate_disk_path(self.diskpath, f".{nameof(obj)}.pt");
                tobj=obj.cpu();
                sharedobj=torch.save(tobj, filepath);
                self.file_manager.add_file(filepath);
                return sharedobj;
            else:
                logging.error(f"Err: denied in all storages. diskspace: {filespaceckeck()}");
            """structure decompose for find inner tensor"""
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._shareallocate(o) for o in obj);
        elif isinstance(obj, dict):
            return {k: self._shareallocate(v) for k, v in obj.items()};
        
        elif isinstance(obj, str):
            """non-tensor convertable object in direct: str"""
            if not storepath:
                sharedobj=torch.tensor(obj.encode("utf-8"), dtype=torch.uint8, device=device);
            else:
                filepath=generate_disk_path(self.diskpath, f".{nameof(obj)}.pt");
                tobj=torch.tensor(obj.encode("utf-8"), dtype=torch.uint8, device=device);
                sharedobj=torch.save(tobj, filepath);
                self.file_manager.add_file(filepath);
            return sharedobj;
        else:
            return ray.get(ray.set(obj));
        
    def _batch_raytask(self, Clsfunc, task_sign, task_inputs):
        """ray batch running. handling resource and types are in workers by _Rayworker class"""
        def _args_bytid(task_inputs, target_taskid, endrange_taskd=None):
            if endrange_taskd is None:
                return next((task for task in task_inputs if task["taskid"] == target_taskid), None);
            else:
                return [task for task in task_inputs if target_taskid <= task["taskid"] <= endrange_taskd];
        
        if batch<0:
            """devided batch processings"""
            wholebatch=self._get_estimate(task_sign) if self._get_estimate(task_sign) is not None else int(ramthreashold/self.cworkers);
            batch=_get_optimal_batch_size(len(task_inputs), wholebatch, len(task_inputs[0]));
        else:
            """limits batch size under output size"""
            batch=min(batch, len(task_inputs));
        """batched tasking"""
        results=[];
        for ind in range(0, len(task_inputs), batch):
            unitbat=_args_bytid(task_inputs, ind, ind+batch);
            results.append([self._worker.collectedres(Clsfunc, args=task["args"], kwargs=task["kwargs"]) for task in unitbat]);
            self._update_local_state();
        return results;
    
    def multiprocess(self, Clsfunc=None, Clsmethod=None, tensorout=False, batch=0, *args, **kwargs):
        """Meta-multiprocessing. workers are self._worker, have Rayactor as meta ray actor. Clsfunc, Clsmethod as ray actor.

        Args:
            Clsfunc (ray.remote class|function): target class or function. ignored after initialization with not None Clsfunc.
            Clsmethod (method of ray.remote class): target method in the Clsfunc if that is class. ignored after initialization with not None Clsmethod.
            tensorout (bool, optional): trigger for tensortype out_. Defaults to False.
            batch (int, optional): task dividing into batches. positive in bytes unit is manual batch size, all negatives means autosized._ Defaults to 0(whole tasks at the same).
            args (any, optional): args. ignored after initialization with args.
            kwargs (any, optional): kwargs. ignored after initialization with kwargs.
        Returns:
            _list|tensor_: _result vars. nn.Module Class itself also changes._
        """
        if self.Clsfunc is not None:
            Clsfunc=self.Clsfunc;
            Clsmethod=self.Clsmethod if self.Clsmethod else None;
        else:
            self._worker=_Rayworker.options(num_cpus=self.cworkers, num_gpus=self.gworkers).remote(self);
        args=self.args if self.args is not None else args;
        kwargs=self.kwargs if self.kwargs is not None else kwargs;
        
        """dynamic multitasking"""
        shared_args=self._shareallocate(args);
        shared_kwargs=self._shareallocate(kwargs);
        """signature set for caching"""
        if isinstance(Clsfunc, (function, ray.actor.ActorHandle)):
            """signature is only for batched process"""
            task_sign=getsignature(Clsfunc);
        else:
            raise TypeError(f"Class/methods without ray wrapper are not processable to ray library");
            
        # Task input을 한 번에 준비
        task_inputs = [{"taskid": len(shared_kwargs)*aid+kid, "args": shared_args[aid], "kwargs": shared_kwargs[kid]} for aid in range(len(shared_args)) for kid in range(len(shared_kwargs))];
        if torch.cuda.is_available():
            """ram usage measurement part 1"""
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            start_vram = total_memory - reserved_memory
        else:
            start_vram=0;
        start_ram = psutil.virtual_memory().used
        # Ray의 병렬 실행
        """ray running. handling resource and types are in workers by _Rayworker class"""
        if batch!=0:
            """devide tasks into unit batches"""
            results=self._batchtask(Clsfunc, Clsmethod, task_sign, task_inputs);
        else:
            """calculate all in the same time"""
            results = self._worker.collectedres(Clsfunc, Clsmethod, task_inputs["args"], task_inputs["kwargs"])
            self._update_local_state();
            #results = ray.get([self._worker.collectedres.remote(taskid=task["taskid"], clsfunc=Clsfunc, args=task["args"], kwargs=task["kwargs"]) for task in task_inputs]);

        """ram usage measure, part 2"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            end_vram = total_memory - reserved_memory
        end_ram = psutil.virtual_memory().used

        ram_used = end_ram - start_ram
        vram_used = end_vram - start_vram if torch.cuda.is_available() else 0
        self._estimate_caching(signature(task_sign), ram_used, vram_used);
        
        logging.info(f"RAM released: {ram_used/(1024**2)}MB");
        logging.info(f"VRAM released: {vram_used/(1024**2)}MB");
        if tensorout:
            results=torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in results if istensorable(r)]);
        """releasing taskspace"""
        self._cleanup();
        return results if results else [];
    
    def _cleanup(self):
        try:
            # Clear Redis client
            self.redis_client.close()
            # Remove temporary files
            self.clear_files()
            # Release RAM and VRAM
            ram_used = self.release_ram()
            vram_used = self.release_vram()
            return ram_used, vram_used;
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    
    def __del__(self):
        try:
            self.redis_client.close();
            self.clear_files();
        except Exception as e:
            logging.error(f"ERR on cleanup {e}");
        finally:
            if ray.is_initialized():
                ray.shutdown();