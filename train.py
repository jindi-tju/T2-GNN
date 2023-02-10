from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from utils import load_data, accuracy
from models import GCN, Teacher_F, Teacher_S
from args import args
from logit_losses import *
from ppr_matrix import topk_ppr_matrix

# Model and optimizer
class Train:
    def __init__(self, args,repeat,acc_fea,acc_str,acc_stu):
        self.args = args
        self.repeat = repeat
        self.best_teacher_fea_val, self.best_teacher_str_val, self.best_student_val = 0, 0, 0
        self.teacher_fea_state,  self.teacher_str_state, self.student_state = None, None, None
        self.load_data()
        self.acc_list_fea = acc_fea
        self.acc_list_str = acc_str
        self.acc_list = acc_stu

        # Model Initialization
        self.fea_model = Teacher_F(num_nodes=self.features.shape[0],
                                 in_size=self.features.shape[1],
                                 hidden_size=self.args.hidden_fea,
                                 out_size=self.labels_oneHot.shape[1],
                                 num_layers=self.args.num_fea_layers,
                                 dropout=self.args.dropout_fea)
        self.fea_model.to(args.device)

        self.str_model = Teacher_S(num_nodes=self.features.shape[0],
                                  in_size=self.features.shape[1],
                                  hidden_size=self.args.hidden_str,
                                  out_size=self.labels_oneHot.shape[1],
                                  dropout=self.args.dropout_str,
                                  device=args.device)
        self.str_model.to(args.device)

        self.stu_model = GCN(nfeat=self.features.shape[1],
                           nhid=self.args.hidden_stu,
                           nclass=self.labels_oneHot.shape[1],
                           dropout=self.args.dropout_stu,
                           nhid_feat=self.args.hidden_fea,
                           nhid_stru=self.args.hidden_str)

        self.stu_model.to(args.device)

        # Setup loss criterion
        self.criterionTeacherFea = nn.CrossEntropyLoss()
        self.criterionTeacherStr = nn.CrossEntropyLoss()
        self.criterionStudent = nn.CrossEntropyLoss()
        self.criterionStudentKD = SoftTarget(args.Ts)

        # Setup Training Optimizer
        self.optimizerTeacherFea = optim.Adam(self.fea_model.parameters(), lr=self.args.lr_fea, weight_decay=self.args.weight_decay_fea)
        self.optimizerTeacherStr = optim.Adam(self.str_model.parameters(), lr=self.args.lr_str, weight_decay=self.args.weight_decay_str)
        self.optimizerStudent = optim.Adam(self.stu_model.parameters(), lr=self.args.lr_stu, weight_decay=self.args.weight_decay_stu)

    def load_data(self):
        # load data
        self.ttadj, self.tadj, self.adj, self.features, self.labels, self.labels_oneHot, self.train_idx, self.val_idx, self.test_idx = load_data(args.dataset, self.repeat,
                                                                                       args.device, args.rate)
        doc_node_indices = list(range(self.ttadj.shape[0]))
        ppr_matrix = topk_ppr_matrix(sp.csr_matrix(self.ttadj, dtype=float), args.alpha_ppr, args.epsilon,
                                     doc_node_indices, args.topk,
                                     keep_nodes=doc_node_indices)
        ppr_matrix = torch.FloatTensor(ppr_matrix.todense()).to(args.device)
        self.tadj = (self.tadj + ppr_matrix).to(args.device) # A+A_ppr

        print('Data load init finish')
        print('Num nodes: {} | Num features: {} | Num classes: {}'.format(
            self.adj.shape[0], self.features.shape[1], self.labels_oneHot.shape[1] + 1))

    def pre_train_teacher_fea(self,epoch):
        t = time.time()
        self.fea_model.train()
        self.optimizerTeacherFea.zero_grad()

        output,_ = self.fea_model(self.features)
        loss_train = self.criterionTeacherFea(output[self.train_idx], self.labels[self.train_idx])
        acc_train = accuracy(output[self.train_idx], self.labels[self.train_idx])
        loss_train.backward()
        self.optimizerTeacherFea.step()

        if not self.args.fastmode:
            self.fea_model.eval()
            output, _ = self.fea_model(self.features)

        loss_val = self.criterionTeacherFea(output[self.val_idx], self.labels[self.val_idx])
        acc_val = accuracy(output[self.val_idx], self.labels[self.val_idx])

        if acc_val > self.best_teacher_fea_val:
            self.best_teacher_fea_val = acc_val
            self.teacher_fea_state = {
                'state_dict': self.fea_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerTeacherFea.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def pre_train_teacher_str(self, epoch):
        t = time.time()
        self.str_model.train()
        self.optimizerTeacherStr.zero_grad()
        output,_ = self.str_model(self.tadj)
        loss_train = self.criterionTeacherStr(output[self.train_idx], self.labels[self.train_idx])
        acc_train = accuracy(output[self.train_idx], self.labels[self.train_idx])
        loss_train.backward()
        self.optimizerTeacherStr.step()

        if not self.args.fastmode:
            self.str_model.eval()
            output,_ = self.str_model(self.tadj)

        loss_val = self.criterionTeacherStr(output[self.val_idx], self.labels[self.val_idx])
        acc_val = accuracy(output[self.val_idx], self.labels[self.val_idx])

        if acc_val > self.best_teacher_str_val:
            self.best_teacher_str_val = acc_val
            self.teacher_str_state = {
                'state_dict': self.str_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch + 1,
                'optimizer': self.optimizerTeacherStr.state_dict(),  # 保留模型和参数
            }
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        
    def train_student(self, epoch):
        t = time.time()
        self.stu_model.train()
        self.optimizerStudent.zero_grad()

        output, middle_emb_stu = self.stu_model(self.adj, self.features)
        soft_target_fea, middle_emb_fea = self.fea_model(self.features)
        soft_target_str, middle_emb_str = self.str_model(self.tadj)
        contrast_fea, contrast_str = self.stu_model.loss(middle_emb_stu, middle_emb_fea, middle_emb_str)
        loss_train = self.criterionStudent(output[self.train_idx], self.labels[self.train_idx]) + self.args.lambd * (self.criterionStudentKD(output, soft_target_fea) + contrast_fea) + (1 - self.args.lambd)*(self.criterionStudentKD(output, soft_target_str) + contrast_str)
        acc_train = accuracy(output[self.train_idx], self.labels[self.train_idx])
        loss_train.backward()
        self.optimizerStudent.step()

        if not self.args.fastmode:
            self.stu_model.eval()
            output, _ = self.stu_model(self.adj, self.features)

        loss_val = self.criterionStudent(output[self.val_idx], self.labels[self.val_idx])
        acc_val = accuracy(output[self.val_idx], self.labels[self.val_idx])

        if acc_val > self.best_student_val:
            self.best_student_val = acc_val
            self.student_state = {
                'state_dict': self.stu_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerStudent.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        
    def test(self, ts='teacher_fea'):
        if ts == 'teacher_fea':
            model = self.fea_model
            criterion = self.criterionTeacherFea
            model.eval()
            output, _ = model(self.features)
            loss_test = criterion(output[self.test_idx], self.labels[self.test_idx])
            acc_test = accuracy(output[self.test_idx], self.labels[self.test_idx])
            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            self.acc_list_fea.append(round(acc_test.item(), 4))
        elif ts == 'teacher_str':
            model = self.str_model
            criterion = self.criterionTeacherStr
            model.eval()
            output,_ = model(self.tadj)
            loss_test = criterion(output[self.test_idx], self.labels[self.test_idx])
            acc_test = accuracy(output[self.test_idx], self.labels[self.test_idx])
            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            self.acc_list_str.append(round(acc_test.item(), 4))
        elif ts == 'student':
            model = self.stu_model
            criterion = self.criterionStudent
            model.eval()

            output, _ = model(self.adj, self.features)

            loss_test = criterion(output[self.test_idx], self.labels[self.test_idx])
            acc_test = accuracy(output[self.test_idx], self.labels[self.test_idx])
            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            self.acc_list.append(round(acc_test.item(), 4))

    def save_checkpoint(self, filename='./.checkpoints/'+args.dataset, ts='teacher_fea'):
        print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            torch.save(self.teacher_fea_state, filename)
            print('Successfully saved feature teacher model\n...')
        elif ts == 'teacher_str':
            torch.save(self.teacher_str_state, filename)
            print('Successfully saved structure teacher model\n...')
        elif ts == 'student':
            torch.save(self.student_state, filename)
            print('Successfully saved student model\n...')
        
        
    def load_checkpoint(self, filename='./.checkpoints/'+ args.dataset, ts='teacher_fea'):
        print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            load_state = torch.load(filename)
            self.fea_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherFea.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded feature teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        elif ts == 'teacher_str':
            load_state = torch.load(filename)
            self.str_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherStr.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded structure teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        elif ts == 'student':
            load_state = torch.load(filename)
            self.stu_model.load_state_dict(load_state['state_dict'])
            self.optimizerStudent.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded student model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())

