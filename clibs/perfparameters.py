import torch;


torch.set_default_dtype(torch.float64);#torch.set_default_device("cpu");
realTType=torch.float64;
compTType=torch.complex128;
#set_up_backend("torch", "float64");
torch.set_printoptions(precision=20);
vErr=1E-5;
cpu_reserve_fraction=0.8;