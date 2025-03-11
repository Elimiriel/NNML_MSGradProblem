import torch;
from .phyparameters import rAdS, G_N, tensordot2dind, ensure_tensordotdim;
from .perfparameters import vErr;
from torchquad import MonteCarlo;

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
            w=ensure_tensordotdim(self.w, z.size(dim=-1));
        else:
            w=self.w;
        if self.a.size(dim=-1)!=z.size(dim=-1):
            a=ensure_tensordotdim(a, z.size(dim=-1));
        else:
            a=self.a;
        z2fz = torch.float_power(z, 2)*torch.tensordot(w, z, dims=tensordotind)+self.b;
        #if torch.any(z2fz>upperlimit):
        #    z2fz=upperlimit;
        return (1.0 - z)*( 1.0 + torch.tensordot((self.a+1.0), z, dims=tensordotind) - z2fz);
    
    def Temperature(self):  # defined by g(z)
        dg = torch.autograd.functional.jacobian(self.metric_G, self.zsinghoriz);#( self.metric_G(one) - self.metric_G(one-eps) ) / eps
        return -dg/(4.0*torch.pi);
    
    def metric_H(self, z): #g(z) in paper
        #upperlimit = 1.0 + torch.tensordot((self.a+1.0), z, dims=tensordotind);
        if self.w.size(dim=-1)!=z.size(dim=-1):
            w=ensure_tensordotdim(self.w, z.size(dim=-1));
        else:
            w=self.w;
        if self.a.size(dim=-1)!=z.size(dim=-1):
            a=ensure_tensordotdim(a, z.size(dim=-1));
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

    def __calc__(self):
        """physical model equation"""
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
    
    def parameters(self):
        """extracts learnable parameters"""
        return self.a, self.w, self.b;