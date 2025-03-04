import torch;
import torch.nn as nn;
from torchquad import MonteCarlo;
#from .tensormultiprocessings import FunctMP, raydeclare;
from .perfparameters import vErr;
from .phyparameters import rAdS, G_N, tensordotind, ensure_tensordotdim;

class Metric_GH():
    def __init__(self, zs, a, w, b):  # Sch. or RN: switch=0, GR: switch=1
        """2nd line of each definitions are tensordot-able converts if defout is scalar"""
        a=a if isinstance(a, torch.Tensor) else torch.tensor(a);
        self.a=a.unsqueeze(0) if self.a.dim()==1 else a;
        w=w if isinstance(w, torch.Tensor) else torch.tensor(w);
        self.w=w.unsqueeze(0) if self.w.dim()==1 else w;
        b=b if isinstance(b, torch.Tensor) else torch.tensor(b);
        self.b=b.unsqueeze(0) if self.b.dim()==1 else b;
        zs=zs if isinstance(zs, torch.Tensor) else torch.tensor(zs);
        self.zs=zs.unsqueeze(0) if zs.dim()==1 else zs;
        self.zsinghoriz=torch.ones_like(zs);
        
    """For gradient tracking, min and max to prevent negative metrics are replaced by if"""
    def metric_G(self, z):#f(z) in paper
        #upperlimit = 1.0 + torch.tensordot((self.a+1.0), z, dims=tensordotind);
        if self.w.size(dim=-1)!=z.size(dim=-1):
            w=enensure_tensordotdim(self.w, z.size(dim=-1));
        else:
            w=self.w;
        if self.a.size(dim=-1)!=z.size(dim=-1):
            a=enensure_tensordotdim(a, z.size(dim=-1));
        else:
            a=self.a;
        z2fz = torch.float_power(z, 2)*torch.tensordot(w, z, dims=tensordotind)+self.b;
        #if torch.any(z2fz>upperlimit):
        #    z2fz=upperlimit;
        return (1.0 - z)*( 1.0 + torch.tensordot((self.a+1.0), z, dims=tensordotind) - z2fz);
    
    def Temperature(self):  # defined by g(z)
        dg = torch.func.jacfwd(self.metric_G)(self.zsinghoriz);#( self.metric_G(one) - self.metric_G(one-eps) ) / eps
        return -dg/(4.0*torch.pi);
    
    def metric_H(self, z): #g(z) in paper
        #upperlimit = 1.0 + torch.tensordot((self.a+1.0), z, dims=tensordotind);
        if self.w.size(dim=-1)!=z.size(dim=-1):
            w=enensure_tensordotdim(self.w, z.size(dim=-1));
        else:
            w=self.w;
        if self.a.size(dim=-1)!=z.size(dim=-1):
            a=enensure_tensordotdim(a, z.size(dim=-1));
        else:
            a=self.a;
        z2fz = torch.float_power(z, 2)*torch.tensordot(w, z, dims=tensordotind)+self.b;
        #if torch.any(z2fz>upperlimit):
        #    z2fz=upperlimit;
        return 1.0 + torch.tensordot(a, z, dims=tensordotind) - z2fz;
    
    def EntropyDensity(self):  # defined by h(z)
        h1 = self.metric_H(self.zsinghoriz);
        return (rAdS**2)*h1/(4.0*G_N);
        
class Integrand(Metric_GH):
    def __init__(self, zs, a, w, b):
        nn.Module.__init__(self);
        Metric_GH.__init__(self, zs, a, w, b);
        
    def _interg_commons(self, zs, t):
        u = t if isinstance(t, torch.Tensor) else torch.tensor(t);
        u=u.unsqueeze(0) if u.dim()==1 else u;
        gzu = self.metric_G(zs*u)
        hzu = self.metric_H(zs*u)
        hz = self.metric_H(zs)
        H=(hzu/hz)**2
        return u, gzu, hzu, H;
        
    def integrand1(self, t):
        #dl
        zs = self.zs;
        u, gzu, hzu, H=self._interg_commons(zs, t);
        H=torch.where(H<u**4, H, H+u**4);
        #if torch.any(H < u**4):
        #    H=u**4;
        return 2.0*zs*(u**2)/( torch.sqrt(gzu*hzu*(H-u**4)) );
    
    def integrand2(self, t):
        #dC
        zs = self.zs;
        u, gzu, hzu, H=self._interg_commons(zs, t);
        H=torch.where(H<u**4, H, H+u**4);
        #if torch.any(H< u**4):
        #    H=u**4;
        return ( torch.sqrt(gzu*hzu*(H-u**4)) - 1.0 )/(u**2);

class Model_Fn(Integrand):
    def __init__(self, zs, a, w, b, integrator=MonteCarlo()):
        Integrand.__init__(self, zs, a, w, b);
        #cworks, gworks=raydeclare(None, cpuR, gpuR)
        #FunctMP.__init__(self, cworkers=cworks, gworkers=gworks);
        self.integrator=integrator;

    def result(self):
        integrand1, integrand2 = self.integrand1, self.integrand2;
        int_init, int_fin = 0.0, 1.0;
        zs = self.zs
        intsampleN=max(zs.size().numel());
        hz = self.metric_H(zs);
        if intsampleN<30:
            intsampleN=vErr**-1;
        integrator=self.integrator
        #sample=N;
        l_m, C_m = [], []
        for i in range(intsampleN):
            l_m.append( integrator.integrate(integrand1, dim=1, N=intsampleN, integration_domain=[[int_init, int_fin]], backend='torch') )
            C_m.append( -1. + integrator.integrate(integrand2, dim=1, N=intsampleN, integration_domain=[[int_init, int_fin]], backend='torch') )
        l_m, C_m = torch.stack(l_m).median(0).values, torch.stack(C_m).median(0).values
        
        """if intsampleN>1000:
            #pararell on each z points
            subdomains=[[start, end] for start, end in zip(torch.linspace(0, zs.max(), intsampleN)[:-2], torch.linspace(0, zs.max(), intsampleN))];
            kwargs1=[{"fn": integrand1, "dim": -1, "N": intsampleN//len(subdomains), "intergration_domain": [domain], "backend": "torch"} for domain in subdomains];
            kwargs2=[{"fn": integrand2, "dim": -1, "N": intsampleN//len(subdomains), "intergration_domain": [domain], "backend": "torch"} for domain in subdomains];
            
        else:
            #pararell on each integrals in one z point
            kwargs1=[{"fn": integrand1, "dim": -1, "N": intsampleN, "intergration_domain": [[0, zs[ind].item()]], "backend": "torch"} for ind in range(len(zs))];
            kwargs2=[{"fn": integrand2, "dim": -1, "N": intsampleN, "intergration_domain": [[0, zs[ind].item()]], "backend": "torch"} for ind in range(len(zs))];
        l_m=self.multiprocessing(self.intergrat, "integrate", kwargs=kwargs1, tensorout=True);
        C_m=self.multiprocessing(self.intergrat, "integrate", kwargs=kwargs2, tensorout=True)-1.0;"""
        S_m = hz*l_m/(2*zs**2) + C_m/zs
        return l_m, S_m

class DNNP(nn.Module):
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
        nn.Module.__init__(self);
        self.a=nn.Parameter(torch.randn(1));
        if not all(isinstance(n, int) and n > 0 for n in nords_eachlayer_list):
            raise ValueError("only positive int")
        if nords_eachlayer_list[0]!=1 | nords_eachlayer_list[-1]!=1:
            raise ValueError("nords of in and out must be 1");
        self.strinfo=nords_eachlayer_list;
        self.activF = activefunction;
        self.initlearnR=learnrate;
        for lev, nodes in enumerate(nords_eachlayer_list):
            if metadendmode is not None:
                """metadendrites"""
                self.LL[lev] = nn.ModuleList(nn.Parameter([torch.randn(nodes, nodes)]));
                """tensordot-able convert if def res is scalar"""
                self.LL[lev]=self.LL[lev].unsqueeze(0) if self.LL[lev].dim()==1 else self.LL[lev];
                if metadendmode:
                    """metabias"""
                    self.LLB[lev]=nn.ModuleList(nn.Parameter([torch.randn(nodes)]));
                    self.LLB[lev]=self.LLB[lev].unsqueeze(0) if self.LLB[lev].dim()==1 else self.LLB[lev];
            """bias of each nodes"""
            self.b[lev]=nn.ModuleList(nn.Parameter([torch.randn(nodes)]));
            """weight of each nodes"""
            self.w[lev]=nn.ModuleList(nn.Parameter([torch.randn(nodes, nodes)]));
        self.modelclass=modelclass if isinstance(modelclass, Model_Fn) else TypeError(f"this model is customized to Model_Fn class structure, else define new task and add to forward.");
        self.zt=z;
        
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
        return resl, resS