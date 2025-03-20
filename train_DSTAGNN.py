#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from time import time
import shutil
import argparse
import configparser
from tqdm import tqdm
from model.wDSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix,load_PA
from lib.utils1 import load_graphdata_channel1, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 允许在服务器环境生成图像
import glob


 
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(1)


parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_dstagnn.conf', type=str,
                    help="configuration file path")
parser.add_argument("-f", "--file", required=False)
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
stag_filename = data_config['stag_filename']
strg_filename = data_config['strg_filename']
print(stag_filename)
print(strg_filename)
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

graph_use = training_config['graph']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
print('batch_size\t', batch_size)

num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = 1
d_model = int(training_config['d_model'])
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
in_channels = int(training_config['in_channels'])
num_of_d = in_channels
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
n_heads = int(training_config['n_heads'])
d_k = int(training_config['d_k'])
d_v = d_k

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('myexperiments', dataset_name, folder_dir)
print('params_path:', params_path)


_, train_loader, train_target_tensor, _, val_loader, val_target_tensor, _, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

adj_TMD = load_weighted_adjacency_matrix(stag_filename, num_of_vertices)
adj_pa = load_PA(strg_filename)

adj_merge = adj_TMD

net = make_model(DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_merge,
                 adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads,patch_size=4)


net = net.to(DEVICE)


def train_main():
    
    # 创建带时间戳的保存目录
    save_dir = f"./training_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化训练记录
    history = {
        'train_loss': [],
        'val_loss': [],
    
    }

    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('nb_block\t', nb_block)
    print('nb_chev_filter\t', nb_chev_filter)
    print('nb_time_filter\t', nb_time_filter)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    

    criterion = nn.SmoothL1Loss().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True,
    min_lr=1e-5
   )
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:
        checkpoint = torch.load(params_filename, map_location=DEVICE)
        net.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])  # 新增

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        #net.load_state_dict(torch.load(params_filename))
        net.load_state_dict(torch.load(params_filename, map_location=DEVICE, weights_only=True)) 

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in tqdm(range(start_epoch, epochs)):
        print('current epoch: ', epoch)
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        val_loss = compute_val_loss_mstgcn(DEVICE, net, val_loader, criterion, sw, epoch)
        print('val loss', val_loss)
        scheduler.step(val_loss)  # 根据验证损失调整学习率

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            #torch.save(net.state_dict(), params_filename)
            torch.save({
            'epoch': epoch,
            'model_state': net.state_dict(),
            'optim_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),  
            'best_val_loss': best_val_loss,  # 保存最佳验证损失
            'best_epoch': best_epoch,        # 保存最佳轮次
            'history': history,              # 保存完整历史记录
            'global_step': global_step,       # 保存全局步数
            'history': history
            }, f'{save_dir}/checkpoint_epoch{epoch}.pth')
            
            history['val_loss'].append(val_loss)
            print('best epoch: ', best_epoch)
            print('best val loss: ', best_val_loss)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        epoch_train_loss = 0
       
        for batch_index, batch_data in enumerate(train_loader):
            
            encoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            #outputs = net(encoder_inputs)
           
            outputs = net(encoder_inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()
            epoch_train_loss += training_loss 

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            encoder_inputs = encoder_inputs.to('cpu')
            labels = labels.to('cpu')

            if global_step % 1000 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    if epoch % 5 == 0:
        # 生成损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.savefig(f'{save_dir}/loss_epoch{epoch}.png')
    
     

    print('best epoch:', best_epoch)
    avg_train_loss = epoch_train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)  # 记录平均训练损失
    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''
    
    #base_path = '/root/autodl-tmp/w-dstagnn/myexperiments/PEMS04/dstagnn_h1d0w0_channel1_1.400000e-04'
    #params_filename = os.path.join(base_path, f'epoch_{global_step}.params')

    #params_filename = os.path.join(model_dir, f'checkpoint_epoch{global_step}.pth')  # 修改后路径
    #params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
   
    model_dir='/root/autodl-tmp/w-dstagnn/training_logs/20250319-184320'
    params_filename = os.path.join(model_dir, f'checkpoint_epoch{global_step}.pth')  # 正确路径格式
    print('load weight from:', params_filename)
    

    checkpoint = torch.load(params_filename, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state'])
    #net.load_state_dict(torch.load(params_filename))
    #net.load_state_dict(torch.load(params_filename, map_location=DEVICE, weights_only=True))

    predict_and_save_results_mstgcn(DEVICE, net, data_loader, data_target_tensor, global_step, _mean, _std, model_dir, type)
    #predict_and_save_results_mstgcn(DEVICE,net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type)
    
if __name__ == "__main__":

    train_main()
    

    #predict_main(10, test_loader, test_target_tensor, _mean, _std, 'test')
    









