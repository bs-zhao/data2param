import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
from sklearn.model_selection import train_test_split
import time
from tabulate import tabulate
import os
import math
import copy
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from .tools import *
from .regression_models import *
from .regv2_gpt_trialtime import *
from .regv2_gpt_subjtrial import RegSubjTrial

class ParameterDecoder:
    def __init__(self, decoder_type="reg",
                 dir_save=None, folder_save=None, name_save=None,
                 iid=1, seed=42):
        
        self.decoder_type = decoder_type
        self.iid = iid
        
        if dir_save == None:
            dir_save = os.getcwd()        

        if folder_save == None:
            folder_save = "ParameterDecoder"
            index = 1
            folder_save2 = folder_save
            while os.path.exists(os.path.join(dir_save, folder_save2)):
                folder_save2 = f"{folder_save}{index}"
                index += 1
            self.dir_save = os.path.join(dir_save, folder_save2)
            print(f"dir_save set by default: {self.dir_save}")
        else:
            self.dir_save = os.path.join(dir_save, folder_save)
            print(f"dir_save set by given: {self.dir_save}")
            
        
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
            
        if name_save == None:
            self.name_save = "parameter_decoder"
        else:
            self.name_save = name_save 
        
        self.seed = seed
        self.dim_input = None
        self.dim_data_subj = None
        self.dim_data_trial = None
        self.len_time = None
        self.dim_data_time = None
        self.dim_output = None        

        # model settings
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None   
        
    def save_instance(self, filename=None):

        """Save instance, ensuring models and states are on CPU"""
        if filename==None:
            filename = f"{self.name_save}.pkl"
        else:
            filename = f"{filename}.pkl"

        with open(os.path.join(self.dir_save, filename), 'wb') as f:
            pickle.dump(self, f)

        print("Instance saved")

    def load_model_state(self, path=None):
        if path==None:
            path = os.path.join(self.dir_save, "best_model_state_dict.pth")

        self.model.load_state_dict(torch.load(path, map_location=self.device))            

    def prepare_datafile(self, data_dir=None, num_file_use=-1, key_data_subj=None, key_data_trial=None, key_data_time=None, key_param='params', param_names_remove=None, batch_size=32, val_split=0.2, seq_input_stratagy=0, code=None): 
        
        self.data_dir = data_dir
        self.key_data_subj = key_data_subj
        self.key_data_trial = key_data_trial
        self.key_data_time = key_data_time
        self.key_param = key_param 
        self.param_names_remove = param_names_remove
        
        if key_data_subj==None:
            self.collate_fn = collate_fn_flat_padmask
        else:
            self.collate_fn = collate_fn_subj_flat_padmask
            
        self.batch_size = batch_size
        self.val_split = val_split
        self.seq_input_stratagy = seq_input_stratagy # 0: flatten all; 1: give each sequential input an independent encoder
        
        print("Preparing datafile ...")

        if  "reg" in self.decoder_type:
            self.files = glob.glob(f"{data_dir}/gen*.pkl")
            if num_file_use>0:
                self.files = self.files[:num_file_use]
            
            if len(self.files)>1:
                self.train_files, self.valid_files = train_test_split(self.files, test_size=self.val_split, random_state=self.seed) 
                            
            info= [
                ["Num files", len(self.files)],
                ["Num validation files", len(self.valid_files)],
                ["Batch size", self.batch_size]
            ]        
            print(tabulate(info, tablefmt="grid")) #headers=["Property", "Value"]
        elif "classify" in self.decoder_type:
            self.train_files = []
            self.valid_files = []
            self.num_classes = len(self.data_dir)
            for i in range(self.num_classes):
                current_data_dir = self.data_dir[i]
                current_files = glob.glob(f"{current_data_dir}/gen*.pkl")
                if num_file_use>0:
                    current_files = current_files[:num_file_use]                
                train_files, valid_files = train_test_split(current_files, test_size=self.val_split, random_state=self.seed) 
                self.train_files.append(list(train_files))
                self.valid_files.append(list(valid_files))
        
        self.prepare_dataset(seq_input_stratagy=self.seq_input_stratagy, code=code)
               
    def prepare_dataset(self, batch_size=None, seq_input_stratagy=None,
                        train_loader=None, valid_loader=None, code=None):
        
        print("Prepareing dataset ...")
        
        if batch_size != None:
            self.batch_size = batch_size
            
        if seq_input_stratagy != None:
            self.seq_input_stratagy = seq_input_stratagy
        
        if train_loader==None:

            if "reg" in self.decoder_type:
                self.train_dataset = FolderDictionaryDataset(self.train_files, self.key_data_subj, self.key_data_trial, self.key_data_time, self.key_param, self.param_names_remove)
                self.valid_dataset = FolderDictionaryDataset(self.valid_files, self.key_data_subj, self.key_data_trial, self.key_data_time, self.key_param, self.param_names_remove)
            elif "classify" in self.decoder_type:
                self.train_dataset = FolderDictionaryDataset_classify(self.train_files, self.key_data_subj, self.key_data_trial, self.key_data_time)
                self.valid_dataset = FolderDictionaryDataset_classify(self.valid_files, self.key_data_subj, self.key_data_trial, self.key_data_time)
            
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)  # IterableDataset
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)  # IterableDataset
        else:
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            
        # get initial infomation        
        tmp = next(iter(self.train_loader))        
        self.dim_input = tmp[0].shape[2]
        if "reg" in self.decoder_type:
            self.dim_output = tmp[2].shape[1]  
        elif "classify" in self.decoder_type:
            self.dim_output = len(self.data_dir)
        if self.key_data_subj==None:  
            self.dim_data_subj = None
        else: 
            self.dim_data_subj = tmp[3].shape[1]
            
        if "reg" in self.decoder_type:
            tmp = self.train_files[0]
        elif "classify" in self.decoder_type:
            tmp = self.train_files[0][0]
        self.param_names, self.dic_datadims_subj, self.dic_datadims_trial, self.datadim_time = get_data_info(tmp, self.decoder_type, self.key_data_subj, self.key_data_trial, self.key_data_time, self.key_param, self.param_names_remove, code=code)
        
        self.dim_data_trial = sum(v[1] for v in self.dic_datadims_trial.values())
        if self.key_data_time:
            self.len_time = self.datadim_time[1]
            self.dim_data_time = self.datadim_time[2]
        
        info= [
            # ["dim_input", self.dim_input],
            ["dim_data_subj", self.dim_data_subj],
            ["dim_data_trial", self.dim_data_trial],
            ["dim_data_time", self.dim_data_time],
            ["len_time", self.len_time],
            ["dim_output", self.dim_output],
            ["input_subj", " ".join([f"\n{k} {v}" for k, v in self.dic_datadims_subj.items()])],
            ["input_trial", " ".join([f"\n{k} {v}" for k, v in self.dic_datadims_trial.items()])],
            ["input_time", f"{self.key_data_time} {self.datadim_time}"],
            ["parameter names", ", ".join(self.param_names) if self.param_names is not None else "None"]  # 处理None的情况
        ]       
        print(tabulate(info, tablefmt="grid")) #headers=["Property", "Value"]        
        
    def inspect_dataset(self):
        return
    
    def adjust_model_complexity(self, n):
        return        
        
    def prepare_model(self, lr=0.0001, post_dropout=0):        
        self.initial_lr = lr
        self.post_dropout = post_dropout
        
        if self.dim_data_trial == None:
            print(f"Please prepare data first!")
            return
        
        if "reg" in self.decoder_type:


            if (self.key_data_subj!=None and 
                self.key_data_trial!=None and 
                self.key_data_time==None) or (
                self.key_data_subj!=None and 
                self.key_data_trial!=None and 
                self.key_data_time!=None and 
                self.seq_input_stratagy==0) or (
                self.key_data_subj!=None and 
                self.key_data_trial==None and 
                self.key_data_time!=None and 
                self.seq_input_stratagy==0):

                self.model = Reg_subjtrial(
                    dim_data_subj=self.dim_data_subj,
                    dim_emb_subj=256,
                    dim_data_trial=self.dim_data_trial,
                    dim_emb_trial=128,
                    dim_output=self.dim_output,
                    post_dropout=self.post_dropout
                )


            elif (self.key_data_subj==None and 
                  self.key_data_trial!=None and 
                  self.key_data_time!=None and                   
                  self.seq_input_stratagy==1):
                
                self.model = Reg_trialtime(
                    dim_data_trial=self.dim_data_trial,
                    len_time=self.len_time,
                    dim_data_time=self.dim_data_time,
                    dim_emb_trial=64,        
                    dim_emb_time=64,
                    dim_emb_trial_time=128,
                    dim_output=self.dim_output,
                    post_dropout=self.post_dropout
                )
                
            else: # pure trial                                
                if self.iid==1:
                    self.model = Reg_flat(self.dim_input, self.dim_output, dim_middle=128, post_dropout=self.post_dropout)
                else:
                    self.model = Reg_flat_niid(self.dim_input, self.dim_output, dim_middle=128, post_dropout=self.post_dropout)
                    
        elif "classify" in self.decoder_type:
            
            self.model = Classify_subjtrial(
                dim_data_subj=self.dim_data_subj,
                dim_emb_subj=256,
                dim_data_trial=self.dim_data_trial,
                dim_emb_trial=128,
                dim_output=self.dim_output
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if "reg" in self.decoder_type:
            if self.decoder_type == "normflow":
                self.loss_fn = self.compute_loss_flow
            else:
                self.loss_fn = torch.nn.MSELoss()
        elif "classify" in self.decoder_type:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            
        # training records
        self.train_losses = []
        self.val_losses = []
        self.total_epochs = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        print(self.model)
        print("Model is set")
        
    def compute_loss_flow(self, z, log_det):
        """
        Compute negative log-likelihood loss.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent variable, shape (batch_size, dim_out)
        log_det : torch.Tensor
            Log of Jacobian determinant, shape (batch_size,)
        
        Returns
        -------
        torch.Tensor
            Mean negative log-likelihood loss
        """
        d = z.size(1)  # Dimension number
        prior_logprob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * d * np.log(2 * np.pi)
        loss = -(prior_logprob + log_det).mean()
        return loss

    def train(self, epochs=50, n_patience=20, plot_train=True, plot_valid=True, device=None):      
        
        
        if self.model== None:
            print(f"Please prepare model first!")
            return
        
        self.device = device
        self.device = self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu')   
        self.model = self.model.to(self.device)
        print(f"Training begin with device: {self.device}, batchsize: {self.batch_size}")        
        
        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0
            all_preds_train, all_targets_train = [], []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.total_epochs}, {epoch+1}/{epochs} Training", leave=False)
            
            for inputs, mask, labels, inputs_subj in pbar:
                inputs, mask, labels, inputs_subj = inputs.to(self.device), mask.to(self.device), labels.to(self.device), inputs_subj.to(self.device)
                
                # 确保分类任务的标签是长整型
                if "classify" in self.decoder_type:
                    labels = labels.long()
                
                self.optimizer.zero_grad()
                
                if self.decoder_type == "diffusion":
                    loss = self.model(inputs_subj, inputs, mask, labels)                    
                    x_T = torch.zeros(labels.size(), dtype=labels.dtype, device=labels.device)
                    outputs = self.model.get_estimation_point(inputs_subj, inputs, mask, x_T)
                    
                    loss.backward()

                elif self.decoder_type == "normflow":                    
                    z, log_det = self.model(inputs_subj, inputs, mask, labels)
                    loss = self.compute_loss_flow(z, log_det)                    
                    loss.backward()
                    
                    del z
                    del log_det
                    
                    z = torch.zeros(labels.size(), dtype=labels.dtype, device=labels.device)
                    outputs = self.model.get_estimation_point(inputs_subj, inputs, mask, z)
                    
                else:
                    if self.key_data_subj==None:
                        outputs = self.model(inputs, mask)
                    else:
                        outputs = self.model(inputs_subj, inputs, mask)                    
                    loss = self.loss_fn(outputs, labels)
                    
                    loss.backward()
                    
                self.optimizer.step()
                total_train_loss += loss.item()
                
                all_preds_train.append(outputs.cpu())
                all_targets_train.append(labels.cpu())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_train_loss / len(all_preds_train)
            self.train_losses.append(avg_train_loss)
            
            if plot_train and "classify" not in self.decoder_type:
                # draw
                all_preds = torch.cat(all_preds_train, dim=0).detach()
                all_targets = torch.cat(all_targets_train, dim=0).detach()
                
                ncols = math.ceil(math.sqrt(self.dim_output))  
                nrows = math.ceil(self.dim_output / ncols)         
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
                axes = axes.flatten()                 
                for i in range(self.dim_output):
                    axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.1)
                    axes[i].plot([all_targets[:, i].min(), all_targets[:, i].max()], 
                                 [all_targets[:, i].min(), all_targets[:, i].max()], 
                                 'r--')
                    axes[i].set_xlabel("Actual")
                    axes[i].set_title(f"{self.param_names[i]}")

                for j in range(self.dim_output, len(axes)):
                    fig.delaxes(axes[j])                
                plt.tight_layout()
                info_text = f"Epoch: {self.total_epochs}, TLoss: {avg_train_loss:.4f}"
                plt.suptitle(info_text, fontsize=14, fontweight="bold", y=1.05) 
                save_path = self.dir_save + f"/scatters/e{self.total_epochs}_train_scatter_pat{self.epochs_without_improvement}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')                             
                # plt.show()                  

                
            # validation
            self.model.eval()
            all_preds_valid, all_targets_valid = [], []
            total_val_loss = 0
            with torch.no_grad():
                
                pbar = tqdm(self.valid_loader, desc=f"Epoch {self.total_epochs}, {epoch+1}/{epochs} Validating", leave=False)
                
                for inputs, mask, labels, inputs_subj in pbar:
                    inputs, mask, labels, inputs_subj = inputs.to(self.device), mask.to(self.device), labels.to(self.device), inputs_subj.to(self.device)

                    if self.decoder_type == "diffusion":
                        loss = self.model(inputs_subj, inputs, mask, labels)
                        x_T = torch.zeros(labels.size(), dtype=labels.dtype, device=labels.device)
                        outputs = self.model.get_estimation_point(inputs_subj, inputs, mask, x_T)
                      
                    elif self.decoder_type == "normflow":
                        
                        z, log_det = self.model(inputs_subj, inputs, mask, labels)
                        loss = self.compute_loss_flow(z, log_det)
                        
                        z = torch.zeros(labels.size(), dtype=labels.dtype, device=labels.device)
                        outputs = self.model.get_estimation_point(inputs_subj, inputs, mask, z)
                        
                    else:
                        if self.key_data_subj==None:
                            outputs = self.model(inputs, mask)
                        else:
                            outputs = self.model(inputs_subj, inputs, mask)
                        loss = self.loss_fn(outputs, labels)
                        
                    total_val_loss += loss.item()
                    
                    all_preds_valid.append(outputs.cpu())
                    all_targets_valid.append(labels.cpu())
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                        
            avg_val_loss = total_val_loss / len(all_preds_valid)
            self.val_losses.append(avg_val_loss)
            
            if plot_valid and "classify" not in self.decoder_type:
                # draw
                all_preds = torch.cat(all_preds_valid, dim=0).detach()
                all_targets = torch.cat(all_targets_valid, dim=0).detach()
                
                ncols = math.ceil(math.sqrt(self.dim_output))  
                nrows = math.ceil(self.dim_output / ncols)         
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)
                axes = axes.flatten()                 
                for i in range(self.dim_output):
                    axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.1)
                    axes[i].plot([all_targets[:, i].min(), all_targets[:, i].max()], 
                                 [all_targets[:, i].min(), all_targets[:, i].max()], 
                                 'r--')                    
                    # Compute n and R
                    n = len(all_targets[:, i])
                    r = np.corrcoef(all_targets[:, i], all_preds[:, i])[0, 1]
                    axes[i].text(0.95, 0.05, f"n={n}\nR={r:.2f}",
                                 transform=axes[i].transAxes, fontsize=10,
                                 ha="right", va="bottom",
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))                    
                    axes[i].set_xlabel("Actual")
                    axes[i].set_title(f"{self.param_names[i]}")

                for j in range(self.dim_output, len(axes)):
                    fig.delaxes(axes[j])                
                plt.tight_layout()
                info_text = f"Epoch: {self.total_epochs}, VLoss: {avg_val_loss:.4f}"
                plt.suptitle(info_text, fontsize=14, fontweight="bold", y=1.05) 
                save_path = self.dir_save + f"/scatters/e{self.total_epochs}_valid_scatter_pat{self.epochs_without_improvement}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')                             
                # plt.show()

            if "classify" in self.decoder_type:
                # 计算训练集准确率
                all_preds_train_cat = torch.cat(all_preds_train, dim=0).detach()
                all_targets_train_cat = torch.cat(all_targets_train, dim=0).detach()
                
                # 将预测概率转换为类别索引
                _, pred_train_classes = torch.max(all_preds_train_cat, dim=1)
                
                # 计算准确率
                train_acc = (pred_train_classes == all_targets_train_cat).float().mean().item()
                
                # 计算验证集准确率
                all_preds_valid_cat = torch.cat(all_preds_valid, dim=0).detach()
                all_targets_valid_cat = torch.cat(all_targets_valid, dim=0).detach()
                
                _, pred_valid_classes = torch.max(all_preds_valid_cat, dim=1)
                valid_acc = (pred_valid_classes == all_targets_valid_cat).float().mean().item()
                
                # 打印准确率信息
                print(f"Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}")
                
                # 可视化准确率
                if plot_train or plot_valid:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Training confusion matrix
                    train_cm = confusion_matrix(all_targets_train_cat, pred_train_classes)
                    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
                    ax1.set_title(f'Training Confusion Matrix\nAccuracy: {train_acc:.4f}')
                    ax1.set_xlabel('Predicted')
                    ax1.set_ylabel('True')
                    
                    # Validation confusion matrix
                    valid_cm = confusion_matrix(all_targets_valid_cat, pred_valid_classes)
                    sns.heatmap(valid_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
                    ax2.set_title(f'Validation Confusion Matrix\nAccuracy: {valid_acc:.4f}')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('True')
                    
                    plt.tight_layout()
                    save_path = self.dir_save + f"/accuracy/e{self.total_epochs}_confusion_matrix_pat{self.epochs_without_improvement}.png"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    # plt.show()
            
            # Check early stopping
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, os.path.join(self.dir_save, f"best_model_state_dict.pth"))
                self.save_instance()
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= n_patience:
                print(f"Early stopping triggered with n_patience = {n_patience}")
                break
            
            end_time = time.time()
            print(f"\rEpoch {self.total_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - time: {end_time - start_time:.2f}s - patience: {self.epochs_without_improvement}")
            self.total_epochs += 1

        self.model.load_state_dict(self.best_model_state)
    
    
    # def save_model(self, path="mlp_model.pth"):
    #     torch.save(self.model.state_dict(), path)
    
    # def load_model(self, path="mlp_model.pth"):
    #     state_dict = torch.load(path, map_location=self.device, weights_only=True)
    #     self.model.load_state_dict(state_dict)
    #     self.model.to(self.device)
    
    def predict(self, files, key_param=None, post_dropout=False):
        """Use the model for prediction, ensuring it's on the correct device"""
        # Copy the model and load best weights
        self.current_model = copy.deepcopy(self.model)
        self.current_model.load_state_dict(self.best_model_state)
        self.current_model = self.current_model.to(self.device)
        
        self.test_dataset = FolderDictionaryDataset(files, self.key_data_subj, self.key_data_trial, self.key_data_time, key_param, self.param_names_remove)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)  # IterableDataset
        self.preds = []
        self.params = []
        
        if post_dropout:
            self.current_model.eval()
        else:            
            self.current_model.train()
        with torch.no_grad():
            for inputs, mask, labels, inputs_subj in self.test_loader:
                inputs, mask, labels, inputs_subj = inputs.to(self.device), mask.to(self.device), labels.to(self.device), inputs_subj.to(self.device)
                outputs = self.current_model(inputs_subj, inputs, mask)        
                
                self.preds.append(outputs.cpu())
                if key_param != None:
                    self.params.append(labels.cpu())
                else:
                    self.params.append(outputs.cpu())

        return torch.cat(self.preds, dim=0), torch.cat(self.params, dim=0)
    
    def plot_loss(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()
