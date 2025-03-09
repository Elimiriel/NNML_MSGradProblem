import re;
import torch;
from .phyparameters import xdist, ywidth, rAdS, G_N, zt, ensure_tensordotdim;
from .perfparameters import realTType, vErr;

class Reference():
    def __init__(self, targetmodel, z, beta, mu):
        """_summary_

        Args:
            targetmodel (_str_): _"name of the model"_
            beta (_"(1D list of)float"_): _"chemical potential"_
            mu (_"(1D list of)float"_): _"momentum relaxation strength"_

        Raises:
            ValueError: _description_
        """
        beta=beta if isinstance(beta, torch.Tensor) else torch.tensor(beta);
        self.beta=beta.unsqueeze(0) if beta.dim()==1 else beta;
        mu=mu if isinstance(mu, torch.Tensor) else torch.tensor(mu);
        self.mu=mu.unsqueeze(0) if mu.dim()==1 else mu;
        if re.match(r"^((none|[\b]|\\0|0\.0|0|false|linear\s?\-?)+(axion?))$", targetmodel, re.IGNORECASE):
            self.Q=None;
        elif re.match(r"^(((gubser\-?\s?rocha|gubser|rocha)+\s?\-?)+(axion?))$", targetmodel, re.IGNORECASE):
            self.Q=self._resQ(self.mu, self.beta);
            if self.Q.dim()==1:
                self.Q=self.Q.unsqueeze(0);
        else:
            raise ValueError(f"Unrecognized targetmodel: {targetmodel}");
        """re.match: 문자열의 시작부터 정규식이 일치하는지 확인합니다. re.IGNORECASE: 대소문자 구별을 무시합니다.
        정규식 예:
            linear\s*axion: linear와 axion 사이에 공백이 있어도 매칭.
            gubster\-rocha|gubster|rocha: "gubster-rocha", "gubster", "rocha"를 매칭.
            첫 번째 조건은 none, [\b], \0, 0.0, 0, false, 또는 linear axion을 매칭.
            두 번째 조건은 gubster-rocha, gubster, 또는 rocha를 매칭.
        """
        self.xdist=xdist;
        self.ywidth=ywidth;
        self.z=z if isinstance(z, torch.Tensor) else torch.tensor(z);
    
    def _resQ(self, mu, beta):
        mu=mu if isinstance(mu, torch.Tensor) and mu.size(dim=-1)>0 else ensure_tensordotdim(mu, 0);
        beta=beta if isinstance(beta, torch.Tensor) and beta.size(dim=-1)>0 else ensure_tensordotdim(beta, 0);
        #res=(-6.0+mu**2+torch.sqrt(((beta**4)) - 18.0*(beta**2)))/6.0;
        #res=(1.0/12.0)*(-6+3*beta**2+((-1)**(mode)*np.sqrt(3.0)*torch.sqrt(12-12*beta**2+3*beta**4+16*mu**2)));
        refQ = torch.tensor([0.0, 0.08584617289886354, 0.12296275162005438, 0.4714965502225489]);
        #if torch.all(mu==beta):
        #    res=[refQ[ind] for ind in range(0, len(mu), 1) if res[ind]!=refQ[ind] else res.item()];
        refbeta=torch.arange(0.0, 2.0, 0.5);
        res=torch.where(beta==refbeta[0], refQ[0], torch.where(beta==refbeta[1], refQ[1], torch.where(beta==refbeta[2], refQ[2], torch.where(beta==refbeta[3], refQ[3], refQ[0]))));
        return res;
        
    def ref_G(self, z):
        z=z if isinstance(z, torch.Tensor) else torch.tensor(z);
        z=z.unsqueeze(0) if z.dim()==1 else z;
        targetdim=int(z.size(dim=-1));
        if self.Q is not None:
            Q=self.Q.expand([-1, targetdim]);
            mu=ensure_tensordotdim(self.mu, targetdim);
            beta=ensure_tensordotdim(self.beta, targetdim);
            res=(1.0 - torch.tensordot(torch.ones_like(Q), z, dims=([-1], [0])))*(1 + torch.tensordot(torch.ones_like(Q), z, dims=([-1], [0])) + 3*torch.tensordot(Q, z) + (1 + 3*torch.tensordot(Q*(1 + Q), torch.ones_like(z), dims=([-1], [0])) - torch.tensordot((torch.float_power(beta, 2.0)/2.0), torch.float_power(z, 2.0), dims=([-1], [0]))))/torch.float_power(1 + torch.tensordot(Q, z, dims=([-1], [0])), 3.0/2.0);
        else:
            res=[]
            for mind in range(0, self.mu.size(dim=-1), 1):
                for bind in range(0, self.beta.size(dim=-1), 1):
                    res.append(1.0- (self.beta[bind]**2)*(z**2) / 2.0 + ((self.mu[mind]*z**2)**2) / 4.0- (1.0 - (self.beta[bind]**2 / 2.0)+ (self.mu[mind]**4) / 4.0)*z**3);
            res=torch.stack(res);
        return res;
        
    def ref_H(self, z):
        z=z if isinstance(z, torch.Tensor) else torch.tensor(z);
        targetdim=int(z.size(dim=-1));
        mu=ensure_tensordotdim(self.mu, targetdim);
        beta=ensure_tensordotdim(self.beta, targetdim);
        if self.Q is not None:
            Q=ensure_tensordotdim(self.Q, targetdim);
            return torch.float_power(1 + torch.tensordot(Q, z, dims=([-1], [0])), 3.0/2.0);
        else:
            #발산항 제거용 처리가 제일 처음항
            #bump=torch.tensor(z!=torch.zeros_like(z), dtype=realTType)*0.5*torch.float_power(z, -1)*self.xmax*torch.sqrt(self.ref_G(z));
            #circled=(-torch.float_power(z, -2)+(1/3)*torch.float_power(z, 3))+bump;
            #유도과정 240904 노트 참조. 온갖 변형은 있었으나 마지막은 2차식이며 아래는 근의공식, 여기서 z=0에서의 발산항 제거조건에서 +항이 제외.
            #res=(1/(-2*circled))*(bump-torch.sqrt(-torch.float_power(bump, 2)-4*circled*(z**6)));
            zastr=torch.full_like(self.ref_G(z), z[-2]);
            zhoriz=torch.full_like(self.ref_G(z), z[-1]);
            zz=z.expand_as(self.ref_G(z));
            """meaningless operations in line +1 and +4 are for keeping grads."""
            zelement=ywidth*torch.sqrt(self.ref_G(z))/(2.0*torch.where(torch.abs(zz)>0, zz, torch.full_like(zz, vErr)*(zz/zz)));
            zAelement=ywidth*torch.sqrt(self.ref_G(zastr))/(2.0*zastr);
            aelement=torch.float_power(zastr, -2.0)-(1.0/3.0)*torch.float_power(zastr, -3.0)+zAelement;
            desc=(zelement**2-4*aelement*zhoriz);
            desc=torch.where(desc>=0, desc, torch.zeros_like(desc));
            res=(zelement-torch.sqrt(desc))/(2.0*aelement)
            res=torch.where(torch.isnan(res), torch.tensor(1.0, device=res.device, dtype=res.dtype), res);
            if len(res.size())>2:
                res=[res[ind, ind2] for ind in range(0, len(mu), 1) for ind2 in range(0, len(mu), 1)];
                res=[res[ind] for ind in range(0, len(self.mu), 1)];
            return res;
    
    def ref_HawkT(self, z):
        zhoriz=torch.ones_like(self.ref_G(z), dtype=realTType);
        return torch.func.jacfwd(self.ref_G(z), randomness="same")(zhoriz)/(4*torch.pi);
        #return torch.autograd.grad(self.ref_G(zt), zt, create_graph=True);
            
    def ref_sden(self, z):
        zhoriz=torch.ones_like(self.ref_H(z), dtype=realTType);
        return ((rAdS**2)/(4*G_N))*self.ref_H(zhoriz)/(zhoriz.expand_as(self.ref_H(zhoriz))**2);
            
        
    """def sdT_sols(self):
        beta=self.beta;
        mu=self.mu;
        Q=self.Q;
        zhoriz=self.zhoriz;
        def __TsQ(beta, Q):
            return ( 6*(1+Q)**2 - beta**2 )/(8*torch.pi*((1+Q)**(3/2)));
        def __TsL(mu, beta):
            return (12 - mu**2 - 2*beta**2)/(16*torch.pi);
        T_sol = self.torchMP(ClsMethoFunc=__TsQ, args=(beta, Q), tensorout=True) if Q is not None else self.torchMP(ClsMethoFunc=__TsL, args=(mu, beta), tensorout=True);
        #4*torch.pi*(1+Q)**(3/2) if Q is not None
        sd_sol =  self.ref_sden();
        return T_sol, sd_sol;
    """
    def sdT_sols(self):
        z=self.z;
        return self.ref_HawkT(z), self.ref_sden(z);