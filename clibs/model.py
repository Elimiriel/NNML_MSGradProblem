import torch;
import torch.nn as nn;
import multiprocessing;
from concurrent.futures import ProcessPoolExecutor;
from tensormultiprocessings import cpuworkers, gpuworkers;
#from .tensormultiprocessings import FunctMP, raydeclare;

def checkParamsProperties(model, sampleIn):
    model.zero_grad();
    tempout=model(sampleIn);
    loss=nn.MSELoss()(tempout, torch.zeros_like(tempout));
    loss.backward();
    paramsgrads=[];
    for params in model.parameters():
        if params.grad is None:
            paramsgrads.append("unknown");
        elif params.grad.dim()==sampleIn.dim():
            paramsgrads.append("weight");
        elif params.grad.dim()==1 and params.grad.shape[0]==tempout.shape[-1]:
            paramsgrads.append("bias");
        else:
            paramsgrads.append("unknown");
    return paramsgrads;

class DNN(nn.Module):
    def __init__(self, z, modelclass, nords_eachlayer_list, learnrate, activefunction=nn.ReLU):
        nn.Module.__init__(self);
        if not all(isinstance(n, int) and n > 0 for n in nords_eachlayer_list):
            raise ValueError("only positive int")
        if nords_eachlayer_list[0]!=1 or nords_eachlayer_list[-1]!=1:
            raise ValueError("nords of in and out must be 1");
        if hasattr(modelclass, "parameters"):
            proplist=checkParamsProperties(modelclass, z);
            self.lParamsLayersW=torch.tensor([[torch.randn_like(params).repeat(nords_eachlayer_list[dind], nords_eachlayer_list[dind+1]) if proplist[dind]=="weight" 
                                else torch.ones(nords_eachlayer_list[dind], nords_eachlayer_list[dind+1]) for params in modelclass.parameters()] for dind in range(0, len(nords_eachlayer_list))]);
            self.lParamsLayersB=torch.tensor([[torch.randn_like(params).repeat(nords_eachlayer_list[dind+1]) if proplist[dind]=="bias" 
                                else torch.zeros(nords_eachlayer_list[dind+1]) for params in modelclass.parameters()] for dind in range(0, len(nords_eachlayer_list))]);
        else:
            self.w = torch.tensor([[nn.Parameter(torch.randn(nords_eachlayer_list[i], nords_eachlayer_list[i+1])) 
                                    for i in range(len(nords_eachlayer_list)-1)] for _ in range(len(modelclass))]);
            self.b = torch.tensor([[nn.Parameter(torch.randn(nords_eachlayer_list[i+1])) 
                                    for i in range(len(nords_eachlayer_list)-1)] for _ in range(len(modelclass))]);
            
        self.strinfo=nords_eachlayer_list;
        self.activF = activefunction();
        self.initlearnR=learnrate;
        self.zt=z;
        self.modelclass = modelclass;
        
    def _layer(self, process, depth):
        outdim=torch.tensor(process).size();
        if len(outdim)>1 and outdim[0]>1:
            if self.lParamsLayers is None:
                res=tuple(torch.sum(torch.matmul(self.w[ind], process), dim=self.w[ind].size())
                    +torch.sum(self.b[ind, depth], dim=self.b[ind, depth].size()) for ind in range(0, int(process.size(dim=0))));
                return res;
            else:
                res=tuple(torch.sum(torch.matmul(self.lParamsLayersW[ind], process), dim=self.lParamsLayersW[ind].size())
                    +torch.sum(self.lParamsLayersB[ind], dim=self.lParamsLayersB[ind].size()) for ind in range(0, int()));
                return res;
        else:
            if self.lParamsLayers is None:
                res=torch.sum(torch.matmul(self.w, process), dim=self.w.size())+torch.sum(self.b[depth], dim=self.b[depth].size());
                return res;
            else:
                res=torch.sum(torch.matmul(self.lParamsLayersW, process), dim=self.lParamsLayersW.size())+torch.sum(self.lParamsLayersB, dim=self.lParamsLayersB.size());
                return res;
            
    def forward(self):
        layer=torch.full(torch.Size(self.modelclass), 0.0, dtype=torch.float64);
        with ProcessPoolExecutor(max_workers=cpuworkers) as executor:
            """runtime including windows"""
            process = list(executor.map(self.modelclass, self.z));
            for depth in range(0, len(self.strinfo)):
                layer=self.activF(self._layer(process, depth));
            executor.shutdown(wait=True);
        return layer;