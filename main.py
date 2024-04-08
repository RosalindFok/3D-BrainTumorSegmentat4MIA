"""
Project for 2024 Spring Medical Image Analysis of UCAS
Task: Segmentation of gliomas in pre-operative MRI scans
     sub-regions considered for evaluation{
        1: NCR(necrotic), NET(non-enhancing tumor)
        2: ED(peritumoral edema)
        4: ET(enhancing tumor)
        0: else
     }
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import NvNet
from dataset import BraTSDataset
from criteria import CombinedLoss
from load_path import hdf5_path_list, config


def setup_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}\n')
    torch.cuda.init() if torch.cuda.is_available() else None
    return device

def main():
    # Device
    device = setup_device()
    # Hyperparameter
    hyperparameter = config['Hyperparameter']
    batch_size = hyperparameter['batch_size']
    learning_rate = float(hyperparameter['learning_rate'])
    original_learning_rate = learning_rate
    epoch = hyperparameter['epoch']
    # Data
    hgg_hdf5_path = [x for x in hdf5_path_list if 'HGG' in x][0]
    train_dataloader = DataLoader(BraTSDataset(hdf5_path=hgg_hdf5_path, tag='train'), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    valid_dataloader = DataLoader(BraTSDataset(hdf5_path=hgg_hdf5_path, tag='valid'), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    test_dataloader  = DataLoader(BraTSDataset(hdf5_path=hgg_hdf5_path, tag='test' ), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    # Network
    seg_outChans = config['Model']['seg_outChans']
    # assert seg_outChans == 3, 'seg_outChans must be 3'
    input_shape = next(iter(train_dataloader))[0].shape[-3:]
    model = NvNet(inChans=4, input_shape=input_shape, seg_outChans=seg_outChans, activation='relu', normalizaiton='group_normalization', VAE_enable=True, mode='trilinear')

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of trainable parametes is {trainable_parameters}.')
    model = model.to(device=device)
    print(model)
    # Loss Function
    criterion = CombinedLoss(k1=0.1, k2=0.1, type='WT')
    get_dice  = CombinedLoss(k1=0, k2=0, type='WT') # when k1=k2=0, the result of CombinedLoss = 1-dice
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Train and Valid
    # train and valid loss in each epoch
    all_train_loss, all_valid_loss = [], []
    all_train_dice, all_valid_dice = [], []
    for ep in range(epoch):
        start_time = time.time()
        # train
        model.train()
        train_loss, train_dice = [], []
        learning_rate = original_learning_rate*((1-ep/epoch)**0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        torch.set_grad_enabled(True)
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
            loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
            train_loss.append(loss.item())
            dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
            train_dice.append(1-dice_loss.item())
            # 3 steps of back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # valid
        model.eval()
        valid_loss, valid_dice = [], []
        with torch.no_grad():
            for img, label in valid_dataloader:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
                loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
                valid_loss.append(loss.item())
                dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
                valid_dice.append(1-dice_loss.item())
               
        all_train_loss.append(round(np.mean(train_loss),6))
        all_valid_loss.append(round(np.mean(valid_loss),6))
        all_train_dice.append(round(np.mean(train_dice),6))
        all_valid_dice.append(round(np.mean(valid_dice),6))
        end_time = time.time()

        print(f'Epoch: {ep}. Train Loss={all_train_loss[ep]}. Valid Loss={all_valid_loss[ep]}. Train dice={all_train_dice[ep]}. Valid dice={all_valid_dice[ep]}. Minutes: {round((end_time-start_time)/60, 3)}')
    
    assert len(all_train_loss) == len(all_valid_loss) == len(all_train_dice) == len(all_valid_dice)

    # Test
    model.eval()
    test_dice = []
    with torch.no_grad():
        for img, label in test_dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
            dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
            test_dice.append(1.0-dice_loss.item())
    print(f'dice = {round(np.mean(test_dice), 6)}')

if __name__ == '__main__':
    main()