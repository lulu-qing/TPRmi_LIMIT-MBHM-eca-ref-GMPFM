from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np

# x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.3,shuffle=True)

#该文用于创建基础阶段和增量阶段所要读取的数据样本和标签

class CWRU_dataset(Dataset):
    def __init__(self, root, train=True, index=None, base_sess=None,):
        if train:
            data = np.load(root + "/" + "train_data_0Hcwru_1_2048.npy")
            label = np.load(root + "/" + "train_label_0Hcwru_1_2048.npy")
        else:
            data = np.load(root + "/" + "test_data_0Hcwru_1_2048.npy")
            label = np.load(root + "/" + "test_label_0Hcwru_1_2048.npy")

        self.data = torch.tensor(data, dtype=torch.float)#所有的样本集合
        
        self.targets = torch.tensor(label, dtype = torch.long)#所有的标签集合

        #如果是基础阶段则调用SelectfromDefault来获取相应的训练集
        if base_sess:
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

        else:  # new Class session
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)

    def __getitem__(self,index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)
    
    
    #基础阶段的数据和标签读取
    def SelectfromDefault(self, data, targets, index):
            data_tmp = []   #临时存储筛选后的数据
            targets_tmp = []  #临时存储筛选后的标签
            for i in index:
                ind_cl = np.where(i == targets)[0]  #i == targets：将当前类别i与所有标签targets逐元素比较，返回布尔数组；np.where()：返回满足条件（即标签等于i）的样本索引
                if data_tmp == []:
                    data_tmp = data[ind_cl]  #取出对应索引的振动信号（形状[n,2048]）
                    targets_tmp = targets[ind_cl]   #取出对应标签（形状[n,]）
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))  #垂直堆叠振动信号数据（扩展样本数）
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))  #水平拼接标签数组

            return data_tmp, targets_tmp

   
   
   #用于根据session_.txt文件所指定的索引，从数据集中​​精确选取指定索引的样本​​，专为增量学习阶段的few-shot训练设计。例如从类别4（session_2）中选取预定义的5个样本。
    def NewClassSelector(self, data, targets, index):
            data_tmp = []  #
            targets_tmp = []
            ind_list = [int(i) for i in index]  #将字符串索引转为整数列表（[120, 453, 789, 322, 655]）
            ind_np = np.array(ind_list)   #转为NumPy数组（np.array([120, 453, 789, 322, 655])）
            index = ind_np.reshape((1,5)) #强制转换为二维数组（shape=(1,5)）
            for i in index:
                ind_cl = i
                if data_tmp == []:
                    data_tmp = data[ind_cl]  #通过NumPy高级索引同时获取5个样本的振动信号
                    targets_tmp = targets[ind_cl]  #通过NumPy高级索引同时获取5个样本的标签
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

            return data_tmp, targets_tmp
    
# class_index = np.arange(4)
# path = "./data"
# trainset = CWRU_dataset(root=path, train=True, index=class_index, base_sess=True)
