import os;
import numpy as np;
#from .tensormultiprocessings import FunctMP, raydeclare;
import torch;
import torch.nn as nn;
# Define a custom nameof function as an alternative
from .perfparameters import nameof;
from .model import DNN;
from .dataiod import DataIO, ParaPrepset, ParaPrepLoad, TensDFrame;
from .timegen import Curtimeout, Tpass;

_epoch_range_limit = 50000;
_show_epoch = 100;
_save_epoch = 10;
_optim_func, learnRate = torch.optim.Adam, 1e-4;
_milestones, _lr_gamma = [1000], 0.1;

class Train():
      def __init__(self, reference_model, learning_model, epoch_range, epoch_printinterval=_show_epoch):
            """_"unit training process"_

            Args:
            reference_model (_"Reference Physical Model class"_): _"physical model"_
            beta (_"numeric list, array, tensor"_): _description_
            mu (_"numeric list, array, tensor"_): _description_
            epoch_range (_"numeric var or list, array, tensor"_): _description_
            epoch_printinterval (_"positive int"_): _"graph-out intervals"_
            """
            if isinstance(learning_model, DNN):
                  self.learning_model=learning_model;
                  self.modelequs=learning_model.modelclass;
            else:
                  raise TypeError("reference_model must be a DNN class instance");
            if isinstance(epoch_range, (list, torch.Tensor, np.ndarray)):
                  self.epochCond=max(epoch_range.tolist() if isinstance(epoch_range, torch.Tensor) else epoch_range);
            else:
                  self.epochCond=epoch_range;
            if self.epochCond<0:
                  raise ValueError("limits of err range or try nums must be positive");
            self.showepoch=epoch_printinterval;
            self.z=reference_model.z;
            self.T_sol, self.sd_sol=reference_model.sdT_sols(self.z);
            self.S_s = reference_model.ref_sden(self.z);#Fitting(l_m,Fitting_Coeff)
            
      def train(self, filepath_incl_ext):
            """_"Actual learning process"_
            
            Args:
                  filepath_incl_ext (_"string"_): _"save path"_
            """
            #declarations of fixed spaces
            Model=self.learning_model;
            z=self.z;
            epochCond=self.epochCond;
            show_epoch=self.showepoch;
            
            optimizer = _optim_func(params=Model.parameters(), lr=learnRate);
            scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=_milestones, gamma= _lr_gamma );
            S_s=self.S_s;
            T_sol=self.T_sol;
            sd_sol=self.sd_sol;
            epoch=0;
            penealtySigns = [1,0,1];
            loss_history=[[], [], [], [], []]; param_history=[];
            loss=2.0;
            l1_loss = nn.L1Loss();
            modelequs=self.modelequs;
            dispsaver=_SaveShow(modelequs);
            l_m=None;
            S_m = None;  # Initialize S_m to ensure it is defined
            while loss>epochCond if epochCond<1 else epoch<epochCond:
                  print('\rTrainig epoch: {:05d}'.format(epoch), end='');
                  l_m, S_m = Model.forward();
                  #changing vals during training
                  l_loss = torch.zeros(1);
                  S_loss = l1_loss(S_m, S_s);
                  T_m = modelequs.Temperature();
                  sd_m = modelequs.EntropyDensity();
                  """learning process"""
                  penelty_T = torch.abs(T_sol - T_m)
                  penelty_s = torch.abs(sd_sol - sd_m)
                  loss = penealtySigns[0]*S_loss + penealtySigns[1]*penelty_T + penealtySigns[2]*penelty_s;
                  optimizer.zero_grad();
                  loss.backward();
                  optimizer.step();
                  scheduler.step();
                  
                  """loss history"""
                  loss_history[0].append(loss.item())
                  loss_history[1].append(l_loss.item())
                  loss_history[2].append(S_loss.item())
                  loss_history[3].append(penelty_T.item())
                  loss_history[4].append(penelty_s.item())
                  
                  param_history.append(Model().parameters())
                  TimeAtRun=Curtimeout();
                  modelparameters=modelequs.parameters().clone().detach();
                  """saving"""
                  savetarget=[epoch, S_m.clone().detach(), sd_m.clone().detach(), T_m.clone().detach(), l_m.clone().detach(), loss_history, modelparameters, learnRate, _lr_gamma];
                  if epoch==0|(epoch+1)%_save_epoch==0:
                        dispsaver.eachsave(epoch, filepath_incl_ext, z, savetarget);
                        
                        """save and display"""
                        if epoch == 0 or (epoch+1) % _show_epoch == 0:
                              dispsaver.showcurrent(epoch, show_epoch, filepath_incl_ext, z, savetarget, TimeAtRun);
                  """last stopper"""
                  if epoch>_epoch_range_limit:
                        return l_m, S_m, modelequs.metric_G(z), modelequs.metric_H(z);
                  epoch=epoch+1;
            return l_m, S_m, modelequs.metric_G(z), modelequs.metric_H(z);  # Ensure l_m is properly used

class _SaveShow():
      def __init__(self, reference_model, cpuR=0.8, gpuR=0.8):
            self.T_phys=reference_model.sdT_sols(reference_model.z)[0];
            self.s_phys=reference_model.sdT_sols(reference_model.z)[1];
            #cpures, gpures=raydeclare(None, cpuR, gpuR);
            #FunctMP.__init__(self, cworkers=cpures, gworkers=gpures);
      #def eachsave(self, epoch, filepath_incl_ext, zth, savetarget):
      #      self.torchMP(self._eachsave, args=(epoch, filepath_incl_ext, zth, savetarget));
      
      def eachsave(self, epoch, filepath_incl_ext, zth, savetarget):
            for ind, var in enumerate(savetarget):
                  TensDFrame(True, True, zth, savetarget[ind]).drawseaborn(kind="scatter", xlabel="AdS_z(tlike)", 
                                                                              hue=nameof(savetarget[ind]), ylabel=f"{savetarget[ind]}, epch:{epoch+1}");
                  savepath=filepath_incl_ext.split(os.extsep)
                  savepath=f"{savepath[0]}{os.sep}{nameof(savetarget[ind])}-{epoch}{os.extsep}{savepath[1]}"
                  
      #def showcurrent(self, epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun):
      #      self.torchMP(self._showcurrent, args=(epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun));
            
      def showcurrent(self, epoch, save_interval, filepath_incl_ext, zth, savetarget, TimeAtRun):
            hrs=Tpass(TimeAtRun).hrs;
            mins=Tpass(TimeAtRun).mins;
            seconds=Tpass(TimeAtRun).secs;
            milisecs=Tpass(TimeAtRun).msecs;
            #show_Fig(Model, epoch, loss_history, param_history, _show_epoch, gs0, hs0, l_m0, S_m0, N, sample, savedir)
            savepath=filepath_incl_ext.split(os.extsep);
            loadtarget=[[None]*epoch]*len(savetarget);
            for ind, var in enumerate(savetarget):
                  for eind in range(0, epoch+1, save_interval):
                        path=f"{savepath[0]}{os.sep}{nameof(savetarget[ind])}-{eind}{os.extsep}{savepath[1]}"
                        loadtarget[ind][eind]=DataIO(savetarget[ind], path).tfread();
                        TensDFrame(True, True, zth, loadtarget[ind][eind]).drawseaborn(kind="scatter", xlabel="AdS_z(tlike)", ylabel=nameof(loadtarget[ind]),
                                                                                    hue=f"{eind+1}");
                        TensDFrame(True, True, savetarget[4][eind], loadtarget[ind][eind]).drawseaborn(kind="scatter", 
                                                                                                xlabel="AdS_z(tlike)", 
                                                                                                ylabel=nameof(loadtarget[ind]),
                                                                                                hue=f"{eind+1}");
                        
            TensDFrame(True, True, savetarget[4][:], savetarget[1][:]).drawseaborn(kind="scatter", xlabel="l", ylabel="S", 
                                                                              hue="Epoch");
            print('\r', end='');
            Fig_data = f"Training epoch: {str(epoch+1).zfill(5)},   Running time: {hrs}:{mins}:{seconds}.{milisecs}\n"\
                              +f"loss: {round(savetarget[5, 0, -1], -4)}, l_loss: {round(savetarget[5, 1, -1], -4)}, S_loss: {round(savetarget[5, 2, -1], -4)}\n"\
                              +f"Temperature: (Model) {round(savetarget[3].item(), -4)}, (True) {round(self.T_phys, -4)}\n"\
                              +f"Entropy density: (Model) {round(savetarget[2].item(), -4)}, (True) {round(self.s_phys, -4)}\n";
            print (Fig_data)