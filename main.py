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

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import NvNet
from dataset import BraTSDataset
from criteria import CombinedLoss, Hausdorff_Distance
from load_path import hdf5_path_list, save_npz_path, config


def setup_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device if not torch.cuda.is_available() else torch.cuda.get_device_name(torch.cuda.current_device())}\n')
    torch.cuda.init() if torch.cuda.is_available() else None
    return device

def change_label_with_tumor_type(label : torch.Tensor, tumor_type : str) -> torch.Tensor:
    # Map segmentation labels to binary labels
    if tumor_type == 'WT':
        threshold = 1
    elif tumor_type == 'TC':
        threshold = 2
    elif tumor_type == 'ET':
        threshold = 4
    else:
        raise ValueError('Invalid type')
    label = torch.where(label >= threshold, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))
    return label


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
    dataset_name = config['Model']['dataset']
    assert dataset_name.upper() == 'HGG' or dataset_name.upper() == 'LGG', 'Dataset name must be HGG or LGG.'
    hdf5_path = [x for x in hdf5_path_list if dataset_name in x][0]
    train_dataloader = DataLoader(BraTSDataset(hdf5_path=hdf5_path, tag='train'), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    valid_dataloader = DataLoader(BraTSDataset(hdf5_path=hdf5_path, tag='valid'), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    test_dataloader  = DataLoader(BraTSDataset(hdf5_path=hdf5_path, tag='test' ), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    # Network
    seg_outChans = config['Model']['seg_outChans']
    tumor_type   = config['Model']['tumor_type']
    VAE_enable   = config['Model']['VAE_enable']
    input_shape  = next(iter(train_dataloader))[0].shape[-3:]
    model = NvNet(inChans=4, input_shape=input_shape, seg_outChans=seg_outChans, activation='relu', normalizaiton='group_normalization', VAE_enable=VAE_enable, mode='trilinear')

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of trainable parametes is {trainable_parameters}.')
    model = model.to(device=device)
    print(model)
    # Loss and Metric
    criterion = CombinedLoss(k1=0.1, k2=0.1)
    get_dice  = CombinedLoss(k1=0, k2=0) # when k1=k2=0, the result of CombinedLoss = 1-dice
    get_hausdorff_distance = Hausdorff_Distance()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    def train_valid(model : torch.nn.Module) -> torch.nn.Module:
        for ep in range(epoch):
            start_time = time.time()
            # train
            model.train()
            train_loss, train_dice, train_hausdorff_distance = [], [], []
            learning_rate = original_learning_rate*((1-ep/epoch)**0.9)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            torch.set_grad_enabled(True)
            for img, label in train_dataloader:
                label = change_label_with_tumor_type(label, tumor_type)
                img, label = img.to(device), label.to(device)
                pred = model(img)
                seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
                loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
                train_loss.append(loss.item())
                dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
                train_dice.append(1-dice_loss.item())
                train_hausdorff_distance.append(get_hausdorff_distance(seg_y_pred, label).item())
                # 3 steps of back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # valid
            model.eval()
            valid_loss, valid_dice, valid_hausdorff_distance = [], [], []
            with torch.no_grad():
                for img, label in valid_dataloader:
                    label = change_label_with_tumor_type(label, tumor_type)
                    img, label = img.to(device), label.to(device)
                    pred = model(img)
                    seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
                    loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
                    valid_loss.append(loss.item())
                    dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
                    valid_dice.append(1-dice_loss.item())
                    valid_hausdorff_distance.append(get_hausdorff_distance(seg_y_pred, label).item())

            end_time = time.time()
            print(f'Epoch: {ep}.    Minutes: {round((end_time-start_time)/60, 3)}')
            print(f'Train Loss={round(np.mean(train_loss),6)}.\tValid Loss={round(np.mean(valid_loss),6)}')
            print(f'Train Dice={round(np.mean(train_dice),6)}.\tValid Dice={round(np.mean(valid_dice),6)}')
            print(f'Train Hausdorff={round(np.mean(train_hausdorff_distance),6)}.\tValid Hausdorff={round(np.mean(valid_hausdorff_distance),6)}')
        return model
    
    def test(model : torch.nn.Module) -> dict[float, float]:
        start_time = time.time()
        # test
        model.eval()
        test_dice, test_hausdorff_distance = [], []
        with torch.no_grad():
            for idx, (img, label) in enumerate(test_dataloader):
                label = change_label_with_tumor_type(label, tumor_type)
                img, label = img.to(device), label.to(device)
                pred = model(img)
                seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
                dice_loss = get_dice(seg_y_pred, label, rec_y_pred, img, y_mid)
                test_dice.append(1.0-dice_loss.item())
                hausdorff = get_hausdorff_distance(seg_y_pred, label)
                test_hausdorff_distance.append(hausdorff.item())

                img=img.cpu().numpy()
                label=label.cpu().numpy()
                seg_y_pred=seg_y_pred.cpu().numpy()
                rec_y_pred=rec_y_pred.cpu().numpy()
                # img.shape = batch_size, 4, 160, 192, 128
                # label.shape = batch_size, 160, 192, 128
                # seg_y_pred.shape = batch_size, seg_outChans=1, 160, 192, 128
                # rec_y_pred.shape = batch_size, 4, 160, 192, 128
                # 4 = [Flair, T1, T1CE, T2]
                np.savez(f'{os.path.join(save_npz_path, str(idx)+".npz")}', img=img, label=label, seg=seg_y_pred, rec=rec_y_pred)
        end_time = time.time()
        print(f'Test: Dice = {round(np.mean(test_dice), 6)}, Hausdorff = {round(np.mean(test_hausdorff_distance), 6)}. Minutes: {round((end_time-start_time)/60, 3)}')
        return {'dice': round(np.mean(test_dice), 6), 'hausdorff': round(np.mean(test_hausdorff_distance), 6)}

    # Task: train or predict. train={train, valid, test}; predict={test}.
    task = config['Type']['task']
    assert task.lower() == 'train' or task.lower() == 'predict', 'Task must be train or predict.'
    save_tag = config['Type']['save_tag']
    tmp = ''.join([dataset_name, '_', tumor_type, '_model'])
    saved_model_path = ''.join([tmp, '.pth']) if not save_tag else ''.join([tmp, '_', str(save_tag), '.pth'])

    if task.lower() == 'train':
        model = train_valid(model)
        dice_hausdorff = test(model)
        # Save the trained model
        checkpoint = {'model': model.state_dict()}
        checkpoint.update(dice_hausdorff)
        torch.save(checkpoint, saved_model_path)
        print(f'Trained model ({saved_model_path}) saved.')
    elif task.lower() == 'predict':
        if os.path.exists(saved_model_path):
            checkpoint = torch.load(saved_model_path)
            model.load_state_dict(checkpoint['model'])  
            dice = checkpoint['dice']
            hausdorff = checkpoint['hausdorff']
            print(f'Predicted: Dice = {dice}, Hausdorff = {hausdorff}')
            _ = test(model)
        else:
            print(f'Model: {saved_model_path} not found, please check its path, save_tag in config.yaml, or train the model first.')
            exit(1)

if __name__ == '__main__':
    main()