# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
#from tensorboardX import SummaryWriter

# import local modules
from .utils import get_time_dif


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(args, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    #writer = SummaryWriter(log_dir=args['log_path'] + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(args['num_epochs']):
        print('Epoch [{}/{}]'.format(epoch + 1, args['num_epochs']))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % args['evaluation_steps'] == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_report, dev_acc, dev_loss = evaluate(args, model, dev_iter, load_model=False)
                if dev_loss < dev_best_loss:
                    print('Current loss[%.4f] < Best loss[%.4f]'%(dev_loss, dev_best_loss))
                    print('Save model state to: %s'%args['save_path'])
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), args['save_path'])
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                #writer.add_scalar("loss/train", loss.item(), total_batch)
                #writer.add_scalar("loss/dev", dev_loss, total_batch)
                #writer.add_scalar("acc/train", train_acc, total_batch)
                #writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > args['require_improvement']:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for %i steps, auto-stopping..."%args['require_improvement']) 
                flag = True
                break
        if flag:
            break
    # export scalar data to JSON for external processing
    #writer.close()

def evaluate(args, model, data_iter, load_model=True):
    if load_model:
        model.load_state_dict(torch.load(args['save_path']))
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for item in data_iter:
            texts = item[0]
            labels = item[1]
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    avg_loss = loss_total / len(data_iter)
    print(metrics.classification_report(labels_all, predict_all, digits=4))
    report_dict = metrics.classification_report(labels_all, predict_all, digits=4, output_dict=True)
    print("Accuracy: %.3f"%acc)
    print("Loss: %.3f"%avg_loss)
    #confusion = metrics.confusion_matrix(labels_all, predict_all)
    return report_dict, acc, avg_loss

def predict(args, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts in data_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    predict_all = predict_all.tolist()
    return predict_all
