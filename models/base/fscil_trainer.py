from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import numpy as np
import time
import torch

from .helper import *
from utils import *
from dataloader.data_utils import *

import sys
import os

# --- 新增：用于将终端输出同时保存到文件的 Logger 类 ---
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
# ----------------------------------------------------

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()

        # --- 新增：接管系统输出，保存实验所有打印进度 ---
        log_path = os.path.join(self.args.save_path, 'experiment_terminal_output.log')
        sys.stdout = Logger(log_path)
        print("==================================================")
        print("实验进度与所有打印信息将同步保存在: ", log_path)
        print("==================================================")
        # -------------------------------------------------

        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            # self.best_model_dict = torch.load(self.args.model_dir)['state_dict']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        # 🚀 新增这一段来支持余弦退火
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label

                # --- 新增：打印当前样本的真实(原始)标签 ---
                if hasattr(train_set, 'class_indices') and train_set.class_indices is not None:
                    print('Actual original labels for this session:\n', train_set.class_indices)
                # ------------------------------------------

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)

                    # === 替换为以下两行安全代码 ===
                    transform_val = getattr(testloader.dataset, 'transform', None)
                    self.model = replace_base_fc(train_set, transform_val, self.model, args)

                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval() # 严格冻结特征提取器

                if hasattr(testloader.dataset, 'transform'):
                    trainloader.dataset.transform = testloader.dataset.transform
                
                # 调用原版的安全增量逻辑，内部会自动处理局部微调
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa = test(self.model, testloader, 0, args, session, validation=False)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                
                # === 强制保存增量模型参数 ===
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

                # =========================================================
                # 新增：在最后一个增量任务结束后，计算各子数据集的最终准确率
                # =========================================================
                if session == args.sessions - 1:
                    print("\n================ 计算 MBHM 各子数据集最终准确率 ================")
                    self.model.eval()
                    
                    dataset_names_list = ['CWRU', 'DIRG', 'HIT', 'IMS', 'JUST', 'MFPT', 'NCEPU', 'PU', 'XJTU']
                    group_correct = {k: 0 for k in dataset_names_list}
                    group_total = {k: 0 for k in dataset_names_list}
                    
                    with torch.no_grad():
                        sample_idx = 0 
                        for batch in testloader:
                            data, label = [_.cuda() for _ in batch]
                            logits = self.model(data)
                            preds = torch.argmax(logits, dim=1)
                            
                            for i in range(len(label)):
                                logical_true = label[i].item()
                                logical_pred = preds[i].item()
                                
                                # 直接从 Dataloader 里拿出当前样本的真实数据集名字
                                current_dataset = testloader.dataset.dataset_names[sample_idx]
                                sample_idx += 1
                                
                                if current_dataset in group_total:
                                    group_total[current_dataset] += 1
                                    # 如果预测的标签和真实标签匹配，就算这个数据集对了一个
                                    if logical_pred == logical_true:
                                        group_correct[current_dataset] += 1
                                        
                    for group_name in dataset_names_list:
                        if group_total[group_name] > 0:
                            acc = (group_correct[group_name] / group_total[group_name]) * 100
                            print(f"{group_name:<5} 准确率: {acc:6.2f}%  ({group_correct[group_name]}/{group_total[group_name]})")
                        else:
                            print(f"{group_name:<5} 准确率: N/A (本次测试中未包含该数据)")
                    print("==================================================================\n")
                # =========================================================

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
