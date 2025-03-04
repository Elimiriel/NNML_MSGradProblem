import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp;
import matplotlib as mpl;
from matplotlib import pyplot as plt, font_manager as fm;
from math import gamma
import pandas as pd;
import seaborn as sns;
#"""Old level multiprocessing: complex in use and restricted to python's limits(Thread is still limited)"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED;#new multiprocessings. Thread for multiIO, Process to multitask, as_comp... to wait
from custompacks import *;#set of main codes
mp.set_start_method("spawn", force=True);
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#extra dl model api: 일단 현재 구동 가능 후 사용해보자...
#import keras;
#from keras import layer;
#from keras import ops;

usingGffam=[f.name for f in fm.fontManager.ttflist if "Pretendard" or "Segoe" in f.name] or None;
if usingGffam and usingGffam.index("Pretendard JP Variable"):
    plt.rcParams["font.family"]="Pretendard JP Variable";
else:
    plt.rcParams["font.family"]="Segoe UI Variable Text";
plt.rc('font', size=15)        # 기본 폰트 크기
plt.rc('axes', labelsize=16)   # x,y축 label 폰트 크기
plt.rc('xtick', labelsize=14)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=14)  # y축 눈금 폰트 크기
plt.rc('legend', fontsize=14)  # 범례 폰트 크기
plt.rc('figure', titlesize=17) # figure title 폰트 크기
#plt.rc("figure", figsize=(5.5, 4));

modeltypes = ['Linear-Axion','Gubser-Rocha-Axion'];
Starttime=Starttime;
#k = 1.0
mu = torch.arange(0.0, 2.0, 0.5);
beta = torch.arange(0.0, 2.0, 0.5);
oQ = [0.08584617289886354,0.12296275162005438,0.4714965502225489]
nords=[1, 5, 5, 5, 1];
z=phyp.zt;
zast=phyp.zastr;
vErr=pp.vErr;

def ensure_tensordotdim(tensor, target_dimssize):
    """
    0차원 텐서를 1차원으로 확장, 1차원 텐서를 target_dimssize에 맞춰 확장.
    """
    print(f"Original tensor shape: {tensor.shape}")
    if tensor.dim() == 0:  # 스칼라일 경우
        tensor = tensor.expand(target_dimssize)  # 1차원으로 변환
    elif tensor.dim() == 1:  # 1차원 텐서일 경우
        tensor = tensor.unsqueeze(-1).expand(tensor.size(0), target_dimssize)  # 2차원으로 확장
    print(f"tensor shape: {tensor.shape}")
    return tensor

def ref_G(z, mu, beta, Q):
    z = z if isinstance(z, torch.Tensor) else torch.tensor(z)
    res=[]
    for ind in range(0, len(beta), 1):
        res.append(1.0- (beta[ind]**2)*(z**2) / 2.0 + ((mu[ind]*z**2)**2) / 4.0- (1.0 - (beta[ind]**2 / 2.0)+ (mu[ind]**4) / 4.0)*z**3);
    return torch.stack(res)
def ref_H(z, mu, beta, Q):
        z=z if isinstance(z, torch.Tensor) else torch.tensor(z);
        if Q is not None:
            targetdim=int(z.size(dim=-1));
            Q=Q.expand([-1, targetdim]);
            return torch.float_power(1 + torch.tensordot(Q, z, dims=([-1], [0])), 3.0/2.0);
        else:
            #발산항 제거용 처리가 제일 처음항
            #bump=torch.tensor(z!=torch.zeros_like(z), dtype=realTType)*0.5*torch.float_power(z, -1)*xmax*torch.sqrt(ref_G(z, mu, beta, Q));
            #circled=(-torch.float_power(z, -2)+(1/3)*torch.float_power(z, 3))+bump;
            #유도과정 240904 노트 참조. 온갖 변형은 있었으나 마지막은 2차식이며 아래는 근의공식, 여기서 z=0에서의 발산항 제거조건에서 +항이 제외.
            #res=(1/(-2*circled))*(bump-torch.sqrt(-torch.float_power(bump, 2)-4*circled*(z**6)));
            ywidth=1.0
            zastr=torch.full_like(ref_G(z, mu, beta, Q), z[-2]);
            zhoriz=torch.full_like(ref_G(z, mu, beta, Q), z[-1]);
            zz=z.expand_as(ref_G(z, mu, beta, Q));
            """meaningless operations in line +1 and +4 are for keeping grads."""
            zelement=ywidth*torch.sqrt(ref_G(z, mu, beta, Q))/(2.0*torch.where(torch.abs(zz)>0, zz, torch.full_like(zz, vErr)*(zz/zz)));
            zAelement=ywidth*torch.sqrt(ref_G(zastr, mu, beta, Q))/(2.0*zastr);
            aelement=torch.float_power(zastr, -2.0)-(1.0/3.0)*torch.float_power(zastr, -3.0)+zAelement;
            desc=(zelement**2-4*aelement*zhoriz);
            desc=torch.where(desc>=0, desc, torch.zeros_like(desc));
            res=(zelement-torch.sqrt(desc))/(2.0*aelement)
            return torch.where(torch.isnan(res), torch.tensor(1.0, device=res.device, dtype=res.dtype), res)

# Call the function
refh=ref_H(z, mu, beta, None)
print(ref_G(z, mu, beta, None).shape)
#print(ref_H(z, mu, beta, None))
reffh=[refh[ind, ind2] for ind in range(0, len(mu), 1) for ind2 in range(0, len(mu), 1)]
print(len(reffh))
datastr=torch.stack([z, reffh[0], reffh[1], reffh[2], reffh[3]]).numpy();
data=pd.DataFrame(datastr.T, columns=["z", "0.0", "0.5", "1.0", "1.5"])
long_data = data.melt(id_vars=["z"], var_name="Legend", value_name="g(z)")

Plot=sns.relplot(long_data, kind="scatter", x="z", y="g(z)", hue="Legend")
#Plot.savefig(os.curdir, "graph.png");
#plt.plot(z, reffh[0], reffh[1], reffh[2], reffh[3])

def trialQ(mu, beta, mode):
    #cond=torch.float_power(mu, 1.0)
    #cond=torch.tensor([1.0 if mu[ind]>0 else 0.0 for ind, _ in enumerate(mu) ]);
    #res=torch.where(mu==0, torch.where(beta==0, np.sqrt(1E-5**2/2), torch.sqrt(beta**2/2)), torch.sqrt( (-2.0*mu**2)+(6-(3*beta**2)-2*mu**2)*cond+12*cond**2+6*cond**3));
    return (1.0/12.0)*(-6+3*beta**2+((-1)**(mode)*np.sqrt(3.0)*torch.sqrt(12-12*beta**2+3*beta**4+16*mu**2)));
    #res=torch.sqrt( (-2.0*mu**2)+(6-(3*beta**2)-2*mu**2)*cond+12*cond**2+6*cond**3);
    return res;
def restoremu(beta, Q):
    Q=Q if isinstance(Q, torch.Tensor) else torch.tensor(Q);
    return torch.sqrt(3*Q*(1+Q)*(1-beta**2/(2*(1+Q)**2)));

test=[]
mucalc=[]
for ind in range(0, 3, 1):
    test.append(trialQ(mu, beta, ind));
    print(test[ind]);
    mucalc.append(restoremu(beta, test[ind]));
    print(mucalc[ind])
print(restoremu(beta, refQ));