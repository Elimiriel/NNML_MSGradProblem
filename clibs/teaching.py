import os;
import numpy as np;
#from .tensormultiprocessings import FunctMP, raydeclare;
from .refdata import Reference;
import torch;
import torch.nn as nn;
from varname import nameof;
from .dataiod import DataIO, ParaPrepset, ParaPrepLoad, TensDFrame;
from .timegen import Curtimeout, Tpass;

epoch_range_limit = 50000;
show_epoch = 100
save_epoch = 10
optim_func, learnRate = torch.optim.Adam, 1e-4;
milestones, lr_gamma = [1000], 0.1

class Train(Reference):
      def __init__(self, physical_model, z, beta, mu, epoch_range, epoch_printinterval=show_epoch):
            """_"unit training process"_

            Args:
            physical_model (_"str or str in simple sturuct"_): _"physical model name"_
            beta (_"numeric list, array, tensor"_): _description_
            mu (_"numeric list, array, tensor"_): _description_
            epoch_range (_"numeric var or list, array, tensor"_): _description_
            epoch_printinterval (_"positive int"_): _"graph-out intervals"_
            """
            Reference.__init__(self, physical_model, z, beta, mu);
            if isinstance(epoch_range, (list, torch.Tensor, np.ndarray)):
                  self.epochCond=max(epoch_range);
            else:
                  self.epochCond=epoch_range;
            if self.epochCond<0:
                  raise ValueError("limits of err range or try nums must be positive");
            self.showepoch=epoch_printinterval;
            self.modelname=physical_model;
            self.z=z;
            
      def train(self, Modeltoteach, filepath_incl_ext):
            """_"Actual learning process"_
            
            Args:
                  Modeltoteach (_""_): _""_
            """
            Model=Modeltoteach;
            z=self.z;
            epochCond=self.epochCond;
            show_epoch=self.showepoch;
            
            optimizer = optim_func(params=Model.parameters(), lr=learnRate);
            scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma= lr_gamma )
            
            T_sol, sd_sol=self.sdT_sols();
            epoch=0;
            penealtySigns = [1,0,1]
            loss_history=[[], [], [], [], []]; param_history=[];
            loss=2.0;
            
            while loss>epochCond if epochCond<1 else epoch<epochCond:
                  print('\rTrainig epoch: {:05d}'.format(epoch), end='')
                  l_m, S_m = Model.forward();
                  S_s = self.ref_sden(z);#Fitting(l_m,Fitting_Coeff)
                  l_loss = torch.zeros(1)
                  S_loss=nn.L1Loss(S_m, S_s);
                  #S_loss = nn.L1Loss()(S_m, S_s)#평균이 계산되며 1/N 반영
                  T_m = Model.Temperature();
                  sd_m = Model.EntropyDensity();
                  penelty_T = torch.abs(T_sol - T_m)
                  penelty_s = torch.abs(sd_sol - sd_m)
                  loss = penealtySigns[0]*S_loss + penealtySigns[1]*penelty_T + penealtySigns[2]*penelty_s

                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  scheduler.step()

                  loss_history[0].append(loss.item())
                  loss_history[1].append(l_loss.item())
                  loss_history[2].append(S_loss.item())
                  loss_history[3].append(penelty_T.item())
                  loss_history[4].append(penelty_s.item())
                  
                  param_history.append(Model().parameters())
                  TimeAtRun=Curtimeout();
                  modelparameters=Model.parameters().clone().detach();
                  """횟수별 병렬실행형 상태저장"""
                  savetarget=[epoch, S_m.clone().detach(), sd_m.clone().detach(), T_m.clone().detach(), l_m.clone().detach(), loss_history, modelparameters, learnRate, lr_gamma];
                  if epoch==0|(epoch+1)%save_epoch==0:
                        SaveShow.eachsave(epoch, filepath_incl_ext, z, savetarget);
                        
                        """지정한 횟수마다 기록 출력(저장 겸함)"""
                        if epoch == 0 or (epoch+1) % show_epoch == 0:
                              SaveShow.showcurrent(epoch, filepath_incl_ext, z, savetarget, TimeAtRun);
                  """for 대신 시도횟수 수동증가, limit 도달시 정지"""
                  if epoch>epoch_range_limit:
                        return l_m, S_m, Model.metric_G(z), Model.metric_H(z);
                  epoch=epoch+1;
            return l_m, S_m, Model.metric_G(z), Model.metric_H(z);

class SaveShow():
      #def __init__(self):
            #cpures, gpures=raydeclare(None, cpuR, gpuR);
            #FunctMP.__init__(self, cworkers=cpures, gworkers=gpures);
      #def eachsave(self, epoch, filepath_incl_ext, zth, savetarget):
      #      self.torchMP(self._eachsave, args=(epoch, filepath_incl_ext, zth, savetarget));
      
      def eachsave(self, epoch, filepath_incl_ext, zth, savetarget):
            for ind, var in enumerate(savetarget):
                  TensDFrame(True, True, zth, savetarget[ind]).drawseaborn(kind="scatter", xlabel="AdS_z(tlike)", ylabel=nameof(savetarget[ind]), epoch=f"{epoch+1}");
                  savepath=filepath_incl_ext.split(os.extsep)
                  savepath=f"{savepath[0]}{os.sep}{nameof(savetarget[ind])}-{epoch}{os.extsep}{savepath[1]}"
                  
      #def showcurrent(self, epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun):
      #      self.torchMP(self._showcurrent, args=(epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun));
            
      def showcurrent(self, epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun):
            hrs=Tpass(TimeAtRun).hrs;
            mins=Tpass(TimeAtRun).mins;
            seconds=Tpass(TimeAtRun).seconds;
            milisecs=Tpass(TimeAtRun).milisecs;
            #show_Fig(Model, epoch, loss_history, param_history, show_epoch, gs0, hs0, l_m0, S_m0, N, sample, savedir)
            savepath=filepath_incl_ext.split(os.extsep);
            loadtarget=[];
            for ind, var in enumerate(savetarget):
                  for eind in range(0, epoch+1, save_interval):
                        path=f"{savepath[0]}{os.sep}{nameof(savetarget[ind])}-{eind}{os.extsep}{savepath[1]}"
                        loadtarget.append(DataIO(loadtarget[ind, eind], path).tfread());
            for ind, var in enumerate(savetarget):
                  for eind in range(0, epoch+1, save_interval):
                        TensDFrame(True, True, zth, loadtarget[ind, eind, :, ..., :]).drawseaborn(kind="scatter", xlabel="AdS_z(tlike)", ylabel=nameof(loadtarget[ind]),
                                                                                    hue=f"{eind+1}");
                  
            for ind in range(1, 4, 1):
                  for eind in range(0, epoch+1, save_interval):
                        TensDFrame(True, True, savetarget[4, :], loadtarget[ind, eind, :, ..., :]).drawseaborn(kind="scatter", xlabel="AdS_z(tlike)", ylabel=nameof(loadtarget[ind]));
            TensDFrame(True, True, savetarget[4], savetarget[1]).drawseaborn(kind="scatter", xlabel="l", ylabel="S");
            print('\r', end='');
            Fig_data = f"Training epoch: {str(epoch+1).zfill(5)},   Running time: {hrs}:{mins}:{seconds}.{milisecs}\n"\
                              +f"loss: {round(savetarget[5, 0, -1], -4)},   l_loss: {round(savetarget[5, 1, -1], -4)},   S_loss: {round(savetarget[5, 2, -1], -4)}\n"\
                              +f"Temperature: (Model) {round(savetarget[3].item(), -4)}, (True) {round(self.sdT_sols()[0], -4)}\n"\
                              +f"Entropy density: (Model) {round(savetarget[2].item(), -4)}, (True) {round(self.sdT_sols()[1], -4)}\n";
            print (Fig_data)