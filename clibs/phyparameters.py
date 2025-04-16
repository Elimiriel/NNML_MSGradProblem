import torch;
u_ini = 0.;int_fin = 1.;#z=0:z=var 사이 적분구간 세분용 보조변수
zt = torch.arange(0.0, 1.0+0.01, 0.01);#torch.linspace(0,1,25)
rAdS=zt[-1];
zastr=zt[-2];
xdist=1.0;
ywidth=1.0;
G_N = 1/(16*torch.pi);
#only for 2d matrix. else, complete one of functions in Unuseds
tensordot2dind=([0], [-1]);

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

