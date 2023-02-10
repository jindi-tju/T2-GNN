import numpy as np
import torch
import time
import os
from train import Train
from args import args
from utils import setup_seed

if __name__ == '__main__':
    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')
    setup_seed(args.seed, torch.cuda.is_available())

    acc_fea = []
    acc_str = []
    acc_stu = []
    repeats = 10
    for repeat in range(repeats):
        print('-------------------- Repeat {} Start -------------------'.format(repeat))

        train = Train(args,repeat,acc_fea,acc_str,acc_stu)
        t_total = time.time()

        # pre-train Feature teacher model
        for epoch in range(args.epoch_fea):
            train.pre_train_teacher_fea(epoch)
        train.save_checkpoint(ts='teacher_fea')

        # pre-train Structure teacher model
        for epoch in range(args.epoch_str):
            train.pre_train_teacher_str(epoch)
        train.save_checkpoint(ts='teacher_str')

        # load best pre-train teahcer models
        train.load_checkpoint(ts='teacher_fea')
        train.load_checkpoint(ts='teacher_str')
        print('\n--------------\n')

        # train student model GCN
        for epoch in range(args.epoch_stu):
            train.train_student(epoch)
        train.save_checkpoint(ts='student')

        ## test teahcer models
        train.test('teacher_fea')
        train.test('teacher_str')

        # test student model GCN
        train.load_checkpoint(ts='student')
        train.test('student')

        print('******************** Repeat {} Done ********************\n'.format(repeat+1))


    print('Result: {}'.format(acc_fea))
    print('Avg acc: {:.6f}'.format(sum(acc_fea) / repeats))
    print('Result: {}'.format(acc_str))
    print('Avg acc: {:.6f}'.format(sum(acc_str) / repeats))
    print('Result: {}'.format(acc_str))
    print('Result: {}'.format(acc_stu))
    print('Avg acc: {:.6f}'.format(sum(acc_stu) / repeats))
    print('\nAll Done!')



