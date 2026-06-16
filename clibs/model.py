from __future__ import annotations
from collections.abc import Sequence
from typing import Callable
import torch
from torch.nn import Module, ReLU, Sequential, Linear, Conv2d, MSELoss
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tensormultiprocessings import cpuworkers, gpuworkers
import abc
from types import Uint, ActivFn
#from .tensormultiprocessings import FunctMP


def checkParamsProperties(model, sampleIn):
    model.zero_grad()
    tempout=model(sampleIn)
    loss=MSELoss()(tempout, torch.zeros_like(tempout))
    loss.backward()
    paramsgrads=[]
    for params in model.parameters():
        if params.grad is None:
            paramsgrads.append("unknown")
        elif params.grad.dim()==sampleIn.dim():
            paramsgrads.append("weight")
        elif params.grad.dim()==1 and params.grad.shape[0]==tempout.shape[-1]:
            paramsgrads.append("bias")
        else:
            paramsgrads.append("unknown")
    return paramsgrads

def findlps(modelclass):
    """
    Assigns the parameters of a model class to a list of parameters.
    """
    if not hasattr(modelclass, "parameters"):
        print("modelclass has no parameters, returning NULL")
        return None, None
    proplist=checkParamsProperties(modelclass, torch.zeros(1))
    nords_eachlayer_list=[params.shape[0] for params in modelclass.parameters()]
    lParamWs=torch.tensor([[torch.randn_like(params).repeat(nords_eachlayer_list[dind], nords_eachlayer_list[dind+1]) if proplist[dind]=="weight" 
                            else torch.ones(nords_eachlayer_list[dind], nords_eachlayer_list[dind+1]) for params in modelclass.parameters()] 
                        for dind in range(len(nords_eachlayer_list))])
    lParamBs=torch.tensor([[torch.randn_like(params).repeat(nords_eachlayer_list[dind+1]) if proplist[dind]=="bias" 
                            else torch.zeros(nords_eachlayer_list[dind+1]) for params in modelclass.parameters()] for dind in range(len(nords_eachlayer_list))])
    return lParamWs, lParamBs
# CNN, RNN, DLinear, NLinear, (Bi)LSTM|GRU, Transformer, vice versa are planned to add


class NNModel(Module):
    def __init__(self, z, modelclass, nords_eachlayer_list: Sequence[Uint], learnrate, activefunction: ActivFn=ReLU()):
        if nords_eachlayer_list[0]!=1 or nords_eachlayer_list[-1]!=1:
            raise ValueError("nords of in and out must be 1")
        Module.__init__(self)
        self.wparams, self.bparams=findlps(modelclass)
        self.strinfo=nords_eachlayer_list
        self.activF = activefunction
        self.initlearnR=learnrate
        self.zt=z
        self.modelclass = modelclass
    @abc.abstractmethod
    def layer(self):
        pass
            
    @abc.abstractmethod
    def forward(self):
        pass
    
def layerFNN(nords_in, nords_out, activefunction: ActivFn=ReLU(), device=None):
    """
    General layer for convolutional neural networks (CNNs).
    """
    return Sequential(
        Linear(nords_in, nords_out, bias=True, dtype=torch.float64, device=device),
        activefunction
    )

def layerCNN(nords_in, nords_out, kernel_size=3, stride=1, padding=0, activefunction: ActivFn=ReLU(), device=None):
    """
    General layer for convolutional neural networks (CNNs): kernel decides the near-grid matrix.
    """
    return Sequential(
        Conv2d(nords_in, nords_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, dtype=torch.float64, device=device),
        activefunction()
    )