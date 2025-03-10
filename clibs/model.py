import torch;
import torch.nn as nn;
import multiprocessing;
#from .tensormultiprocessings import FunctMP, raydeclare;
from .phyparameters import tensordot2dind, ensure_tensordotdim;

class DNNSimpleDend(nn.Module):
    def __init__(self, z, modelclass_outputfunc, nords_eachlayer_list, learnrate, learning_params=None, activefunction=nn.ReLU):
        nn.Module.__init__(self);
        if not all(isinstance(n, int) and n > 0 for n in nords_eachlayer_list):
            raise ValueError("only positive int")
        if nords_eachlayer_list[0]!=1 or nords_eachlayer_list[-1]!=1:
            raise ValueError("nords of in and out must be 1");
        if learning_params is None:
            self.w = [[nn.Parameter(torch.randn(nords_eachlayer_list[i], nords_eachlayer_list[i+1])) for i in range(len(nords_eachlayer_list)-1)] for _ in range(len(modelclass_outputfunc))];
            self.b = [[nn.Parameter(torch.randn(nords_eachlayer_list[i+1])) for i in range(len(nords_eachlayer_list)-1)] for _ in range(len(modelclass_outputfunc))];
        else:
            self.learnLParams=[[learning_params[varind].unsqueeze(nords_eachlayer_list[dind], nords_eachlayer_list[dind+1])] for varind in range(0, len(learning_params)) for dind in range(0, len(nords_eachlayer_list))];
        self.strinfo=nords_eachlayer_list;
        self.activF = activefunction;
        self.initlearnR=learnrate;
        self.zt=z;
        self.modelclass_outputfunc = modelclass_outputfunc;
        
    def _layer(self, input, depth):
        if self.learnLParams is None:
            return torch.tensordot(self.w[depth], input, dims=(-1, -1))+self.b[depth];
        else:
            return torch.tensordot(self.learnLParams[0], input, dims=([:, -1, :], -1))+self.learnParams[1];
        
    def forward(self):
        with multiprocessing.Pool() as pool:
            process = pool.starmap(self.modelclass_outputfunc, self.z);
            for rind in range(0, len(self.modelclass_outputfunc)):
                for depth in range(0, len(self.strinfo)):
                    layer=self.activF(self._layer(process, depth));
        return layer;

class DNNCustDend(DNNSimpleDend):
    """_"DNN, modified to fit with asymtopic physical model, Model_Fn"_

    Args:
        DNN (_nn.Module_): _"learning Module with dendrite input matrices layer and function have tuple output"_
    """
    def __init__(self, z, modelclass, nords_eachlayer_list, learnrate, activefunction=nn.ReLU, metadendmode=True):
        """_"DNN, modified to fit with asymtopic physical model, Model_Fn"_

        Args:
            nords_eachlayer_list (_"positive int list"_): _"nords to each layer. start and end must be 1"_
            modelclass (_"class Model_Fn|func"_): _"physical model equ. must have kwargs a as parameter, w as weight, b as bias"_
            learnrate (_float_): _"learning rate"_
            activefunction(_function_): activation function. Default is nn.RuLU.
            metadends (_bool|None_): True for metadends and metabias for whole equ, False for metadends only, None for weight only for in-equ.
        """
        DNNSimpleDend.__init__(self, z, modelclass, nords_eachlayer_list, learnrate, activefunction);
        self.LL=[]
        self.LLB=[]
        for lev, nodes in enumerate(nords_eachlayer_list):
            self.LL[lev] = nn.Parameter(torch.randn(nodes, nodes));
            """tensordot-able convert if def res is scalar"""
            self.LL[lev]=self.LL[lev].unsqueeze(0) if self.LL[lev].dim()==1 else self.LL[lev];
            if metadendmode:
                """metabias"""
                self.LLB[lev]=torch.randn(nodes);
                self.LLB[lev]=self.LLB[lev].unsqueeze(0) if self.LLB[lev].dim()==1 else self.LLB[lev];
        
    def _conventional(self, layer, weight, bias):
        return torch.tensordot(weight, layer, dims=tensordotind)+bias;
    
    def _weightconnect(self, weight1, weight2):
        return torch.tensordot(weight1, weight2, dims=tensordotind);

    def _DunittaskA(self):
        """using LL as main dendrites, without metabias"""
        #layer 단일변수가 개별 레이어 결과저장 역할을 대신함. ind=depthind
        for ind in range(0, len(self.strinfo), 1):
            self.w[ind]=self._weightconnect(self.w[ind-1], self.w[ind]) if ind>0 else self.w[ind];
            layerclass=self.modelclass(self.z, self.a, self.w[ind], self.b[ind], self.cpuR, self.gpuR);
            equ=torch.stack(layerclass.result()[0], layerclass.result()[1], dim=0);
            layer1=self.activF(self._weightconnect(self.LL[ind], equ));
        return layer1[0], layer1[1];

    def _DunittaskB(self):
        """using LL as main dendrites, with metabias"""
        #layer 단일변수가 개별 레이어 결과저장 역할을 대신함. ind=depthind
        layerclass=self.modelclass(self.z, self.a, self.w[0], self.b[0], self.cpuR, self.gpuR);
        initequ=torch.stack(layerclass.result()[0], layerclass.result()[1], dim=0);
        delattr(self.w[1:-1], self.b[1:-1]);
        for ind in range(0, len(self.strinfo), 1):
            layer2=self.activF(self._conventional(self.LL[ind], initequ, self.LLB[ind]));
        return layer2[0], layer2[1];
    
    def _Sunittask(self):
        """using w only as dendrites"""
        layerclass=self.modelclass(self.z, self.a, self.w[0], self.b[0], self.cpuR, self.gpuR);
        equ=torch.stack(layerclass.result()[0], layerclass.result()[1], dim=0);
        #layer 단일변수가 개별 레이어 결과저장 역할을 대신함. ind=depthind
        layer=self.activF(equ);
        for ind in range(1, len(self.strinfo), 1):
            #self.w[ind]=torch.tensordot(self.w[ind-1], self.w[ind], dims=tensordotind);
            layer=self.activF(self._conventional(self.w[ind], layer, self.b[ind]));
        return layer[0], layer[1];

    def forward(self):
        if self.LL:
            if self.LLB:
                resl, resS=self._DunittaskB();
            else:
                resl, resS=self._DunittaskA();
        else:
            resl, resS=self._Sunittask();
        return resl, resS;
    
