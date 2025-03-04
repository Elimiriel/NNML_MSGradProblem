"""torch.multiprocessing은 윈도에선 새 프로세스를 만든 후 데이터를 할당하는 방법만 가능해, 복사를 지원하지 않는 주피터 노트북 사용 불가. 
해당부분을 .py 모듈로 뺀 후, if __name__=="name": 식으로 '메인 블록에서 실행' 해야 가능하나, 이마저도 주피터에선 추천하지 않는 듯함."""

"""GPT 한마디: __getargs 내의 각 매개변수(param)는 함수의 매개변수 정의일 뿐, 실제 값이 아닙니다.
        이로 인해 multiprocessing.Process에 직접적으로 적합하지 않으며, 또한 이 구조에서는 인덱싱과 join 단계에서 문제를 일으킬 수 있습니다.
        프로세스 생성 비용: ProcessPoolExecutor는 멀티프로세스를 생성하여 작업을 분배하므로, 실제 연산보다는 프로세스 시작 및 종료에 더 많은 자원이 소모됩니다. 
        따라서 단순히 nn.Parameter와 같은 속성 탐색에서는 비효율적"""


"""병렬처리 중 데이터 직렬화 관련
        concurrent.futures.ThreadPoolExecutor나 ProcessPoolExecutor는 작업을 병렬화하는 데 유용합니다. 메서드를 호출할 때, 데이터만 전달하여 병렬로 처리합니다.
        PyTorch의 torch.multiprocessing은 nn.Module과 잘 호환되며, GPU 텐서를 병렬 처리하는 경우 특히 유용합니다.
        비교시, 코드 복잡성 외엔 torch쪽이 우위
        """
"""왜 torch.multiprocessing이 GIL 문제를 회피할 수 있는가?
            프로세스 기반 병렬 처리:
            Python의 GIL은 단일 인터프리터 내의 스레드에서만 적용됩니다. 그러나 torch.multiprocessing은 독립적인 프로세스를 생성하기 때문에 각 프로세스는 별도의 GIL을 가지며 병렬 실행이 가능합니다.
            데이터 공유 최적화:
            torch.multiprocessing은 PyTorch 텐서를 프로세스 간 공유 메모리(shared memory) 에 저장할 수 있어, 데이터를 복사하지 않고 병렬 처리할 수 있습니다. 이를 통해 데이터 전달 비용을 줄이고, 병렬 작업의 성능을 높일 수 있습니다.
        GPU와의 호환성:
            PyTorch는 torch.multiprocessing과 GPU 작업의 호환성을 보장합니다. GPU 텐서도 병렬 처리 시 복사되지 않고, 효율적으로 전달됩니다.
            torch.multiprocessing 사용이 적합한 경우
                CPU 병렬 처리:
                    데이터 전처리, 모델 평가 등 CPU에서 수행되는 작업을 병렬화할 때 효율적입니다.
                GPU 기반 모델 병렬화:
                    하나의 GPU를 여러 프로세스에서 병렬로 사용하거나, 여러 GPU에 작업을 나누어 처리할 때 적합합니다.
                큰 데이터를 처리할 때:
                    PyTorch 텐서를 공유 메모리를 통해 전달하기 때문에, 복사 비용 없이 대규모 데이터를 효율적으로 처리할 수 있습니다.
        Quene, Manager 통합으로 인한 비사용코드            
        def __torchQworker(self, methodORfunc, result_quene, *argsdata):
            result=methodORfunc(*argsdata);#실제 연산
            result_quene.put(result);#계산결과를 공유메모리로
    def torchMPQuene(self, methodORfunc=None, *dbatches):
        mp.set_start_method("spawn", True);#짖렬화 문제 방지 및 호환성 관련 초기설정
        res_quene=mp.Queue();#공유공간
        processres=[];
        if hasattr(self, "func") and hasattr(self, "run"):
            methodORfunc=self.func;
            dbatches=self.args;
        
        for batchind in dbatches:
            process=mp.Process(target=self.__torchQworker, args=(methodORfunc, res_quene, batchind));
            processres.append(process);
            process.start();
        
        #병렬실행 종합
        for process in processres:
            process.join();
        res=[res_quene.get() for _ in dbatches];
        return res;
        """

"""
        Queue
            간단한 데이터 전달:
            프로세스 간에 데이터를 교환하는 가장 기본적인 방법.
            FIFO(First In, First Out) 방식으로 동작하며, 데이터를 추가할 때 put, 가져올 때 get을 사용합니다.
            빠른 성능:
            Queue는 성능이 더 빠르지만, 텍스트나 텐서와 같은 단순 데이터에 적합합니다.
            데이터 공유 제한:
            데이터는 직렬화(serialize)된 형태로 전달되며, 공유 메모리에 대한 직접 액세스는 제공하지 않습니다.
        Manager
            공유 객체 관리:
            프로세스 간에 공유 상태를 관리하는 데 사용됩니다. 예를 들어, 공유 딕셔너리, 리스트 등을 만들 수 있습니다.
            데이터를 동기화하고 업데이트해야 하는 경우에 적합합니다.
            다양한 데이터 구조 지원:
            공유 리스트, 딕셔너리, 네임스페이스 등 다양한 데이터 구조를 제공합니다.
            속도 저하:
            Manager는 데이터 동기화를 위해 더 많은 오버헤드가 발생하므로, 성능이 Queue보다 느릴 수 있습니다.
        """
class FunctMP():
    def __init__(self, func=None, *args):
        """_"Multiprocess function runner, GPT 한마디: 클래스의 메소드는 직렬화가 불가능하므로 이 클래스 상속 후 클래스 내부에서 사용해야 함, @staticmethod는 self와의 연결을 끊어 학습용으로 사용불가
        Due to limitations about methods in classes, unserializable from outsode, herit this class and use multiprocess method.
        for global funcs, filling initial func and args"_
        """
        if func is not None:
            self.func=func;
            self.args=args if args else self.__getargs(func);
            #self.nnparams={i: arg for i, arg in enumerate(args) if isinstance(arg, nn.Parameter)} or None;
        
    def __getargs(self, func):
        """Retrieve arguments"""
        if callable(func):
            try:
                sign = inspect.signature(func);
                res={name: param.default for name, param in sign.parameters.items()};
                return res;
            except ValueError:
                return self.args;#람다함수일 경우 이미 전달된 args를 반환
        else:
            print(f"{func} is not callable");
            return {};
    
        
    def __worker(self, clsmethofunc, resultspace, workerid, lock, *argsdata, **kwargs):
        try:
            with lock:
                if isinstance(clsmethofunc, nn.Module):
                    clsmethofunc=clsmethofunc.share_memory();
                resultspace[workerid]=clsmethofunc(*argsdata, **kwargs);
        except Exception as e:
            print(f"Worker {workerid} error: {e}");
    
    def torchMP(self, *args, ClsMethoFunc=None, workernums=workers, dictout=False, **kwargs):
        """Pararell using shared memoryspace by torch

        Args:
            ClsMethoFunc (class|method|function, CodditionalOptional): target object. not optional to in-class use. Defaults to None.
            workernums (positive int, optional): nums of pararells. Defaults to workers.
            dictout (bool, optional): switch for outtype as dictionary. Defaults to False.

        Returns:
            list|dict|torch.Tensor: results
        """
        mp.set_start_method("spawn", force=True);
        if hasattr(self, "func") and hasattr(self, "args"):
            ClsMethoFunc=self.func;
            args=self.args;
            
        with mp.Manager() as manager:
            lock=manager.Lock();#텐서 동기화용 락
            if dictout:
                shared=manager.dict();
            elif any(isinstance(arg, torch.Tensor) for arg in args)|any(isinstance(kargs, torch.Tensor) for kargs in kwargs):
                shared=torch.zeros(workernums, *args[0].shape).share_memory_();
            else:
                shared=manager.list();
            process=[];
            for workerid in range(workernums):#작업분배
                p=mp.Process(target=self.__worker, args=(ClsMethoFunc, shared, workerid, lock, *args), kwargs=kwargs);
                process.append(p);
                p.start();
                
            for p in process:#병렬결과 병합
                p.join();
        return shared;
class ComplexDiff(FunctMP):
    """_"Differential, Complex space considered .res is output. Deprecated: use torch.func.jacfwd or torch.func.jacrev for 1st order, torch.func.hessian for 2nd order"_

    Args:
        ComplexDiff (_function, dict including tensor or array, str or array or tensor, float under abs 0.1_): function with parameters, 
        the axis differential operates on, differential intarval of the axis have differential axis matching(varkey) and input vars(zaxis, dict struct) of the function
    """
    def __init__(self, originalFunct, diffAxis, dAxis=None):
        """_numeric differentials_

        Args:
            originalFunct (_function_): _func to diff_
            diffAxis (_list|np.array|torch.Tensor_): _numeric diff axis_
            dAxis (_float64_, optional): _dz_. Defaults to None(autocalc from diffAxis).

        Raises:
            ValueError: _inaccurate result because of dz_
        """
        FunctMP.__init__(self);
        #현재 호출된 함수의 식
        self.__funequ=originalFunct;
        if dAxis is None:
            dAxis=torch.mean(torch.abs(torch.diff(diffAxis)));
        else:
            self.__dAxis=dAxis;
        if abs(dAxis)>=0.1:
            raise ValueError("Too large interval to accurate differential.");
        self.diffAxis=diffAxis;
        self.zaxis=diffAxis;
        if hasattr(diffAxis, "imag"):
            self.res=self.torchMP(self.diffAxis, ClsMethoFunc=self.compdfn);
        else:
            self.res=self.torchMP(self.diffAxis, ClsMethoFunc=self.realdfn);
    
    def __funcequ(self, zaxis):
        #입력타입을 함수로 하기 위한 처리부
            if isinstance(self.__funequ, function):
                return self.__funequ(zaxis);#analytical possible
            else:
                return self.__funequ;
                #only numerics
    def realdfn(self, z):
        # x(실수) 방향 편미분 역할을 겸함
        zp=z+self.__dAxis;
        f=self.__funcequ(z)
        #f = self.__funcequ(zp);
        if isinstance(self.__funequ, function):
            #analytical possible
            fxp=self.__funcequ(zp);
        else:
            fxp=torch.cat([self.__funequ[1:-2], torch.tensor((self.funequ[-1]+(self.__funequ[-1]-self.__funequ[-2])/self.__dAxis))], -1);
            #when only numerical possible
        #함수타입으로 출력하기 위한 후처리부
        def __outtofunct(fxp):
            return (fxp-f)/self.__dAxis;
        res=__outtofunct(fxp);
        return res;
        
    def compdfn(self, z):
        zaxis=z;
        f = self.__funcequ(zaxis);

        # y(허수) 방향 편미분
        #indvar_y[varkey] = indvar_y[varkey] + 1j * self.__dAxis;
        def __dfdy(z):
            zp=z+self.__dAxis*1.0j
            fyp=self.__funcequ(zp);
            res=(fyp-f)/self.__dAxis;
            return res;

        # Cauchy-Riemann 조건 확인
        dx_re = self.realdfn(zaxis).real#실-실 미분
        #실-허 미분
        dx_im = self.realdfn(zaxis).imag if torch.abs(self.realdfn(zaxis).imag)>vErr else torch.zeros_like(dx_re);

        #허-허 미분
        dy_im = __dfdy(zaxis).imag;
        #허-실 미분
        dy_re = __dfdy(zaxis).real if torch.abs(__dfdy(zaxis).real)>vErr else torch.zeros_like(dy_im);

        #출력을 함수타입으로 하도록 변경
        if torch.allclose(dx_re, dy_re, atol=vErr) and torch.allclose(dx_im, -dy_re, atol=vErr):
            def __funcpdA(variable):
                res=[self.__funequ(variable+self.__dAxis), self.__funequ(variable+1.0j*self.__dAxis)];
                return res;
            return ((__funcpdA[0]-self.__funcequ)+(__funcpdA[1]-self.__funequ))/self.__dAxis;
        else:
            raise ValueError(f"The function is not differentiable at some point(s).");
        
with ThreadPoolExecutor(max_workers=workers) as io_exec, ProcessPoolExecutor(max_workers=workers) as calc_exec:
        io_futures=[];
        calc_futures=[];
        for ind, var in enumerate(modeltypes):
            if isinstance(getattr(Models[ind], "modelequ"), (str, torch.Tensor)):
                io_futures.append(io_exec.submit(Trainprocess[ind].train(Models[ind], z, DataIO(Models[ind], f"{var}"+"-Output"+os.pathsep+"results.txt").path, integrator)));
            else:
                calc_futures.append(calc_exec.submit(Trainprocess[ind].train(Models[ind], z, DataIO(Models[ind], f"{var}"+"-Output"+os.pathsep+"results.txt").path, integrator)));
                        
            # Wait for all futures to complete
            done, _ = wait(io_futures + calc_futures, return_when=FIRST_COMPLETED);
            #출력종합
            """wait 함수를 통해 io_futures와 calc_futures를 하나로 묶어 관리합니다.
            return_when=FIRST_COMPLETED 옵션을 사용하여 어떤 future가 먼저 완료되든 wait가 반환되도록 하여, future.result()을 호출할 때 다른 Executor에 남아 있는 작업에 영향을 주지 않도록 합니다.
            완료된 future의 결과를 results에 추가하고, 나머지 future는 기다리지 않으므로 자원을 효율적으로 사용할 수 있습니다."""
            for future in done:
                  trainresults.append(future.result());
l_m, S_m, f_m, g_m=[], [], [], [];
for ind, var in enumerate(trainresults):
    l_m[ind], S_m[ind], f_m[ind], g_m[ind] = trainresults[ind];
print('Training is done')
Diag=[None]*len(modeltypes);
message=[None]*len(modeltypes);
with ProcessPoolExecutor(max_workers=workers) as calc_exec:
    for ind, var in enumerate(modeltypes):
        Diag[ind], message[ind]=calc_exec.submit(SGHDiag(zast, Models[ind].forward()).lfits());
for ind, var in enumerate(modeltypes):
    print(message[ind]);
    
    def dytfrw(self, size, shared=True, dtype=None, layout=torch.strided, device="cpu", pin_memory=False):
        """_file read and sync to ram's changes_

        Args:
            size (_int_): _"all element numbers, without structure info."_
            shared (bool, optional): _"sync file and ram feature."_ Defaults to True.
            dtype (_type_, optional): _"dtypes of elements."_ Defaults to None for global settings.
            layout (_type_, optional): _"desired layout."_ Defaults to torch.strided.
            device (str, optional): _"target device."_ Defaults to "cpu.
            pin_memory (bool, optional): _"pin to ram"_. Defaults to False.

        Returns:
            _tensor_: _"tensor from file, with sync"_
        """
        path=self.path;
        #with open(path, 'r') as f:
        return torch.from_file(path, shared=shared, size=size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory);
    

"""이하 workers 레거시 코드
            CPU는 시스템 메모리와 스왑 영역을 공유하기 때문에, 메모리 부족 상황을 피하기 위해 데이터를 직렬화(torch.save)하고 필요할 때만 메모리로 불러오게 설계
                    buffer=io.BytesIO();
                    torch.save(clsmethofunc(*inputspace["args"], **inputspace["kwargs"]), buffer);#serialization. to reduce ram usage, replace buffer with filepath
                    resultspace[workerid]=buffer.getvalue();
                elif resultspace=="vram":
                    resultspace[workerid]=torch.tensor(clsmethofunc(*inputspace["args"], **inputspace["kwargs"]), device="cuda");
                else:
                    swap=os.path.join(swapstorage, f"{nameof(clsmethofunc)}-{workerid}_{retry}.tmp.pt");
                    torch.save(clsmethofunc(*inputspace["args"], **inputspace["kwargs"]), swap);
                    resultspace[workerid]=swap;#to read, requires torch.load(path)"""
                    
@ray.remote(num_cpus=workers, num_gpus=torch.cuda.device_count())
class FunctMP():
    def __init__(self, func=None, *args):
        """_"Multiprocess function runner, GPT 한마디: 클래스의 메소드는 직렬화가 불가능하므로 이 클래스 상속 후 클래스 내부에서 사용해야 함, @staticmethod는 self와의 연결을 끊어 학습용으로 사용불가
        Due to limitations about methods in classes, unserializable from outsode, herit this class and use multiprocess method.
        do not call multiprocessing methods directly global in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite.
        and for only global funcs, filling initial func and args
        torch.save and torch.load were used-gpu tensor should be moved to cpu"_
        """
        self.ramthreashold=2**(10*3);#bytes
        if func is not None:
            self.func=func;
            self.args=args if args else self.__getargs(func);
            #self.nnparams={i: arg for i, arg in enumerate(args) if isinstance(arg, nn.Parameter)} or None;
        #self.temp_storage=find_fastest_storage();
        if hasattr(self, "__getargs"):
            delattr(self, "__getargs");
        
    def __getargs(self, func):
        """Retrieve arguments"""
        if callable(func):
            try:
                sign = inspect.signature(func);
                res={name: param.default for name, param in sign.parameters.items()};
                return res;
            except ValueError:
                return self.args;#람다함수일 경우 이미 전달된 args를 반환
        else:
            print(f"{func} is not callable");
            return {};
    
    def _shareallocate(self, obj):
        device=find_fastest_storage();
        if device==swapstorage:
            storepath=os.path.join(swapstorage, os.sep, f"{nameof(obj)}", os.sep);
            
        if isinstance(obj, (torch.Tensor, nn.Module)):
            if not storepath:
                sharedobj=obj.to(device).share_memory_();
            else:
                sharedobj=ParaPrepset(obj, storepath=storepath);
            return sharedobj;
        #"""structure decompose for find inner tensor"""
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._shareallocate(o) for o in obj);
        elif isinstance(obj, dict):
            return {k: self._shareallocate(v) for k, v in obj.items()};
        #non-tensor convertable object
        else:
            res=ray.put(obj);
            return ray.get(res);
            
    def _worker(self, taskid, clsmethodfunc, args, kwargs):
        id=uuid.uuid4();
        try:
            result=clsmethodfunc(*args, **kwargs);
            return taskid, result;
        except Exception as e:
            logging.basicConfig(f"{id}-work_{taskid}.log", level=logging.ERROR);
            logger=logging.getLogger();
            logger.error(f"{taskid}_RAM usage: {check_available_storage()[0] / (1024 * 1024):.2f} MB\n");
            logger.error(f"{taskid}_VRAM usage: {check_available_storage()[1] / (1024 * 1024):.2f} MB\n");
            logger.error(f"Worker {taskid} error: {e}");
            print(f"Worker {taskid} error: {e}");
            return taskid, e;
    
    def torchMP(self, ClsMethoFunc=None, args=(), kwargs={}, workernums=workers, dictout=False, tensorout=False):
        """_Pararell runner for Class, nn.Module, methods, functions_

        Args:
            ClsMethoFunc (_Class|method|function_, NOT optional for method): _target to pararellize_. Defaults to None.
            args (tuple, NOT optional for method): _pararell vars_. Defaults to ().
            kwargs (dict, NOT optional for method): _non-pararell vars_. Defaults to {}.
            workernums (_int>0_, optional): _nums of pararell ops_. Defaults to workers.
            dictout (bool, optional): _trigger for dicttype out_. Defaults to False.
            tensorout (bool, optional): _trigger for tensortype out_. Defaults to False.

        Returns:
            _list|dict|tensor_: _result vars. nn.Module Class itself also changes._
        """
        ramthreashold=self.ramthreashold;
        if dictout and tensorout:
            raise ValueError("choose only one type to output");
        if hasattr(self, "func") and hasattr(self, "args"):
            ClsMethoFunc=self.func;
            args=self.args;
        
        def __typeout(resultout):
            if tensorout:
                if all(isinstance(r, torch.Tensor) for r in resultout):
                    resultout=torch.stack(resultout);
                else:
                    raise ValueError("tensor type|size mismatch");
            elif dictout:
                resultout=dict(zip(range(len(resultout)), resultout));
            return resultout;
        """ram and vram usage checking"""
        ramspace=check_available_storage()[0];
        vramspace=check_available_storage()[1];
        """OOM on ram and vram: filebase process, avoiding torch.save and torch.load due to security"""
        
        if (ramspace<ramthreashold and vramspace<vramthreshold):
            try:
                dataset=ParaPrepset(args, kwargs, preprocess=ClsMethoFunc);
                dataload=ParaPrepLoad(dataset);
                resultout=[];
                for data in dataload:
                    resultout.extend(data);
                del dataload;
                return __typeout(resultout);
            except:
                pass;
        """dynamic multitasking"""
        with mp.Manager() as manager:
            shared_args=self._shareallocate(args);
            shared_kwargs=self._shareallocate(kwargs);
            shared_ClsMethoFunc=self._shareallocate(ClsMethoFunc);
            taskQ=mp.Quene();
            #resQ=manager.Queue();
            rescollector=Collector.remote();
            lock=manager.Lock();#텐서 동기화용 락
            """Assign tasks"""
            for i, (args, kwargs) in enumerate(zip(shared_args, shared_kwargs)):
                taskQ.put((i, shared_ClsMethoFunc, args, kwargs));
            """workers running"""
            process=[];
            for _ in range(workernums):#작업분배
                p=mp.Process(target=self._worker_coordinator, args=(taskQ, rescollector, lock));
                process.append(p);
                p.start();
                
            for p in process:#병렬결과 병합
                p.join();
                p.terminate();
        #restore from serialized
        #restoring str after on-cpu is recommended
        resultout=[];
        while not resQ.empty():
            tid, resE=resQ.get();
            if isinstance(resE, bytes):
                #on-ram data case
                resultout.append(torch.load(io.BytesIO(resE), weights_only=True));
            elif isinstance(resE, torch.Tensor):
                #vram and tensor
                resultout.append(resE.cpu());
            elif swapstorage in resE:
                resultout.append(torch.load(resE, weights_only=True));
            else:
                raise ValueError(f"outputspace item err: {type(resE)}");
        # GPU 작업이 모두 끝난 후 동기화 및 메모리 해제
        torch.cuda.synchronize();
        torch.cuda.empty_cache();
        return __typeout(resultout);
from torch.utils.data import Dataset, DataLoader;
@ray.remote(num_cpus=workers, num_gpus=torch.cuda.device_count())
class Collector():
    def __init__(self):
        self.results=[(int, torch.Tensor)];
        self.diskpath=os.path.join(swapstorage, os.sep, str(uuid.uuid4()), os.sep, str(uuid.uuid5(uuid.NAMESPACE_DNS, "results")), ".pt");
        
    def process(self, tid, tres):
        device=find_fastest_storage();
        self.results.append((tid, tres.to(device)));
        if all(check_available_storage()<(ramthreashold, vramthreshold)):
            self._disksave();
        
    def _disksave(self):
        if not os.path.exists(self.diskpath):
            with open(self.diskpath, "wb") as f:
                torch.save(self.results, f);
        else:
            with open(self.diskpath, "ab") as f:
                torch.save(self.results, f);
        self.results=[];
            
    def resout(self):
        if os.path.exists(self.diskpath):
            with open(self.diskpath, "rb") as f:
                results=torch.load(f, weights_only=True);
        results.append(self.results);
        if check_available_storage()[1]<vramthreshold:
            torch.cuda.empty_cache();
        self.results=[];
        return results;
    

        
class ParaPrepset(Dataset):
    def __init__(self, *data, storepath=swapstorage, preprocess=None):
        """multiprocessed Dataset. do not call multiprocessing directly in windows and jupyter. wrap with if __name__=="__main__", else child process generate same as parent infinite.

        Args:
            storepath (_storagepath_, optional): _path to save_. Defaults to swapstorage.
            preprocess (_class|method|func_, optional): _preprocessing tasks_. Defaults to None.
        """
        Dataset.__init__(self);
        self.data = data;
        self.obj=preprocess;
        self.storepath=storepath;
        os.makedirs(self.storepath, exist_ok=True);
        self._preprocess();
        self.id=uuid.uuid4();
        
    def _preprocess(self):
        # 데이터를 전처리하는 함수
        def tofile(idx, item):
            filename=os.path.join(self.storepath, f"d{self.id}-{idx}.tmp.pt");
            if not os.path.exists(filename):
                if self.obj:
                    processed=self.obj(item);
                else:
                    processed=item;
                torch.save(processed, filename);
                
        Parallel(n_jobs=workers)(delayed(tofile)(idx, item) for idx, item in enumerate(self.data));
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 데이터 병렬 전처리 수행
        filename=os.path.join(self.storepath, f"d{self.id}-{idx}.tmp.pt");
        if os.path.exists(filename):
            return torch.load(filename, weights_only=True);
        else:
            raise FileNotFoundError(f"{filename} not exist");
    
class ParaPrepLoad(DataLoader):
    def __init__(self, data, loadlocation="cpu", batches=64, workers=workers):
        """multiprocessed Dataloader

        Args:
            data (Dataset|ParePreset): _reading target_
            loadlocation (str, optional): _cpu|cuda_. Defaults to "cpu".
            batches (int, optional): _nums of batches_. Defaults to 64.
            workers (int, optional): _nums of cores_. Defaults to workers.
        """
        pindevice=torch.device(loadlocation);
        pinmemory= loadlocation!="cpu";
        DataLoader.__init__(self, data, batch=batches, num_workers=workers, pin_memory=pinmemory, pin_memory_device=pindevice);
        
class NNLP(nn.Module):
    #size=modelclass.size() if hasattr(modelclass, "size") else len(modelclass);
    #if list(size)!=list(layer.size()):
            #    print(f"modelout size:{layer.size()}!=modelin size:{size}");
            #if len(list(size))>1:
            #    layer=layer.view(size);
            
    def unused():
        """
    def matdmul_tdotind(tensor1, tensor2):
        dim1=-1;
        dim2=0;
        dim1list=[indx1 for indx1 in range(dim1, -tensor1.dim()-1, -1)];
        dim2list=[indx2 for indx2 in range(dim2, tensor2.dim()+1, 1)];
        if len(dim1list)==len(dim2list):
            return (dim1list, dim2list);
        else:
            print(f"axis nums don't match: 1: {len(dim1list)}, 2: {len(dim2list)}");
            return ValueError(f"axis nums don't match: 1: {len(dim1list)}, 2: {len(dim2list)}");
    """
    return 0;
@ray.remote
class Rayworker():
    def __init__(self, ResourceManager, Clsfunc, Clsmethod=None):
        """Ray Actor Prototype. must declare actual unit task Clsfunc.

        Args:
            ResourceManager (_ResourceManager_): ResourceManager. should share with metatask.
            Clsfunc (ray wrapped Class|function): actual task. must be @ray.remote.
            Clsmethod (_Clsfunc.method_): actual task in Clsfunc as a method. Default is None.
        """
        self.taskname=None;
        self.savefilepath=None;
        self.runid=str(uuid.uuid4()).join(str(uuid.uuid5(uuid.NAMESPACE_DNS, "results")));
        self.ResourceManager=ResourceManager;
        if not (hasattr(Clsfunc, "__ray_actor_class__") or hasattr(Clsfunc, "__ray_function__")):
            raise TypeError("Clsfunc must be a @ray.remote wrapped class or function.")
        self.Clsfunc=Clsfunc;
        if hasattr(Clsfunc, "__ray_actor_class__") and Clsmethod is not None:
            self.resmethod=Clsfunc.Clsmethod;
        
    def workerbase(self, taskid, args, kwargs):
        """in actual work, 'self._worker' ray Actor is generated from this, and the Actor is actual worker"""
        """ignore_signals for subporcess works be controlled by the main"""
        self.ResourceManager.ignore_signals();
        try:
            """interrupt handler handling: subprocess ignores and controlled by the main"""
            device=find_fastest_storage();
            if hasattr(self, "resmethod"):
                result=self.resmethod(args, kwargs);
            else:
                result=self.Clsfunc(args, kwargs);
            
            """typeconvert"""
            if not istensorable(result):
                result=torch.tensor(result.encode("utf-8"), dtype=torch.unit8);
            elif istensorable(result):
                result=torch.tensor(result);
                
            """Processing"""
            if device!=swapstorage:
                return taskid, result.to(device);
            else:
                """saving return path"""
                self.savefilepath=self.saveres(result, swapstorage, taskid);
                return taskid, None;
            
        except Exception as e:
            """interrupt handler handling: subprocess ignores and controlled by the main"""
            worker_elog(taskid, e);
            print(f"Worker {taskid} error: {e}");
            return taskid, e;
        
    def saveres(self, res, diskpath, tid):
        self.ResourceManager.ignore_signals();
        savepath=os.path.join(diskpath, f"{self.runid}-{self.taskname}-{tid}.pt");
        torch.save(res.cpu(), savepath);
        return savepath;
        
    def savepath(self):
        self.ResourceManager.ignore_signals();
        return self.savefilepath;

    def fileload(self, taskid):
        self.ResourceManager.ignore_signals();
        if os.path.exists(self.savefilepath):
            try:
                return torch.load(self.savefilepath, weights_only=True);
            except Exception as e:
                logging.error(f"ERR loading worker results {taskid}: {e}");
                return e;
        else:
            """no except return: no datatouch if path is None"""
            pass;