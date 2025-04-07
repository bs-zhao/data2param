import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import pickle
import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')

DATATYPE = torch.float32

def collate_fn_flat_padmask(batch, padding_value=-1):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [len(x) for x in inputs] 
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    masks = torch.zeros_like(padded_inputs[..., 0])
    for i, length in enumerate(lengths):
        masks[i, :length] = 1.0 
    
    label =torch.stack(labels)
    return padded_inputs, masks, label, label

def collate_fn_subj_flat_padmask(batch, padding_value=-1):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    inputs_subj = [item[2] for item in batch]
    lengths = [len(x) for x in inputs] 
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    masks = torch.zeros_like(padded_inputs[..., 0])
    for i, length in enumerate(lengths):
        masks[i, :length] = 1.0 
    
    # 将整数标签转换为张量
    if isinstance(labels[0], int):
        label = torch.tensor(labels, dtype=torch.long)
    else:
        label = torch.stack(labels)
        
    # 处理subjects输入
    if inputs_subj[0] is None:
        inputs_subj = torch.zeros(len(inputs), 1)  # 创建一个填充张量
    else:
        inputs_subj = torch.stack(inputs_subj)
        
    return padded_inputs, masks, label, inputs_subj

def collate_fn_classify_padmask(batch, padding_value=-1):
    """为分类任务特别设计的collate函数，处理整数类标签"""
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]  # 这些是整数类别标签
    inputs_subj = [item[2] for item in batch]
    lengths = [len(x) for x in inputs] 
    
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=padding_value)
    masks = torch.zeros_like(padded_inputs[..., 0])
    for i, length in enumerate(lengths):
        masks[i, :length] = 1.0 
    
    # 将整数类别标签转换为张量
    label = torch.tensor(labels, dtype=torch.long)
    
    # 处理subjects输入
    if inputs_subj[0] is None:
        inputs_subj = torch.zeros(len(inputs), 1)  # 创建一个填充张量
    else:
        inputs_subj = torch.stack(inputs_subj)
        
    return padded_inputs, masks, label, inputs_subj

class FolderDictionaryDataset(IterableDataset):
    def __init__(self, files, key_data_subj=None, key_data_trial=None, key_data_time=None, key_param='params', param_names_remove=None, code=None):
        self.files = files
        self.key_data_subj= key_data_subj
        self.key_data_trial = key_data_trial
        self.key_data_time = key_data_time
        self.key_param = key_param
        self.param_names_remove = param_names_remove      
        self.code = code
        
    # def __len__(self):
    #     return len(self.files)

    def parse_file(self, file):
        with open(file, "rb") as f:
            dic = pickle.load(f)
            for sample in dic:      
                if self.key_param != None:
                    
                    if self.code!=None:
                        exec(self.code)
                        
                    # remove un_wanted params and sort
                    if self.param_names_remove!=None:
                        dict_params = {k: v for k, v in sorted(sample[self.key_param].items()) if k not in self.param_names_remove}
                    else:
                        dict_params = {k: v for k, v in sorted(sample[self.key_param].items())}
                    
                    params = torch.tensor(list(dict_params.values())).to(DATATYPE)
                else:
                    params = torch.tensor([-1]).to(DATATYPE)
                    
                X = torch.cat([torch.tensor(sample[key]).clone().detach() for key in self.key_data_trial], dim=1).to(DATATYPE)
                
                if self.key_data_time != None:
                    X = torch.cat([X, torch.tensor(sample[self.key_data_time]).reshape(X.shape[0], -1).clone().detach()], dim=1).to(DATATYPE)                
               
                if self.key_data_subj==None:                
                    X_subj = None
                else:
                    X_subj = torch.cat([torch.tensor(sample[key]).view(-1).clone().detach() for key in self.key_data_subj], dim=0).to(DATATYPE)
                
                yield X, params, X_subj

    def __iter__(self):
        random.shuffle(self.files)
        for file in self.files:
            yield from self.parse_file(file)

class FolderDictionaryDataset_classify(IterableDataset):
    def __init__(self, file_list, key_data_subj=None, key_data_trial=None, key_data_time=None):
        self.files_list = file_list
        self.num_classes = len(file_list)
        self.key_data_subj = key_data_subj
        self.key_data_trial = key_data_trial
        self.key_data_time = key_data_time
        
        # 确认所有类别的文件数量相同
        file_counts = [len(files) for files in file_list]
        if len(set(file_counts)) > 1:
            print(f"Warning: Not all classes have the same number of files: {file_counts}")
        
        self.max_files = max(file_counts)

    def parse_files(self):
        # 对每个文件索引位置
        for file_idx in range(self.max_files):
            # 加载每个类别对应索引的文件
            samples_by_class = []
            
            for label, files in enumerate(self.files_list):
                # 如果当前类别文件数不足，跳过
                if file_idx >= len(files):
                    continue
                    
                file_path = files[file_idx]

                with open(file_path, "rb") as f:
                    dic = pickle.load(f)
                    class_samples = []
                    
                    for sample in dic:
                        X = torch.cat([torch.tensor(sample[key]) for key in self.key_data_trial], dim=1).to(DATATYPE)
                        
                        if self.key_data_time != None:
                            X = torch.cat([X, sample[self.key_data_time].reshape(X.shape[0], -1)], dim=1).to(DATATYPE)                
                    
                        if self.key_data_subj==None:                
                            X_subj = None
                        else:
                            X_subj = torch.cat([torch.tensor(sample[key]).view(-1) for key in self.key_data_subj], dim=0).to(DATATYPE)
                        
                        class_samples.append((X, label, X_subj))
                    
                    samples_by_class.append(class_samples)

            # 将所有类别的样本拼接成一个列表
            all_samples = []
            for class_samples in samples_by_class:
                all_samples.extend(class_samples)
            
            # 打乱所有样本的顺序
            random.shuffle(all_samples)
            
            # 输出打乱后的样本
            for sample in all_samples:
                yield sample

    def __iter__(self):
        yield from self.parse_files()

            
def get_data_info(file, decoder_type="reg", key_data_subj=None, key_data_trial=None, key_data_time=None, key_param='params', param_names_remove=None, code=None):
    with open(file, "rb") as f:
        dics = pickle.load(f)
        sample = dics[0]        
        if code!=None:
            exec(code)
        
        if "reg" in decoder_type:
            # remove un_wanted params and sort
            if param_names_remove!=None:
                dict_params = {k: v for k, v in sorted(sample[key_param].items()) if k not in param_names_remove}
            else:
                dict_params = {k: v for k, v in sorted(sample[key_param].items())}            
            param_names = list(dict_params.keys())
        else:
            param_names = None
        dic_datadims_trial = {}
        dic_datadims_subj = {}
        for k in key_data_trial:
            dic_datadims_trial[k] = list(sample[k].shape)
            
        if key_data_subj != None:            
            for k in key_data_subj:
                dic_datadims_subj[k] = list(sample[k].shape)
            
        if key_data_time != None:
            datadim_time = list(sample[key_data_time].shape)
            if len(datadim_time)==2:
                datadim_time.append(1)
        else:
            datadim_time = None
        
        return param_names, dic_datadims_subj, dic_datadims_trial, datadim_time