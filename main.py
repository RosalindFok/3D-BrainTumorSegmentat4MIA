"""

"""
import h5py
import torch
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
    learning_rate = hyperparameter['learning_rate']
    epoch = hyperparameter['epoch']
    # Data
    hgg_hdf5_path = [x for x in hdf5_path_list if 'HGG' in x][0]
    dataloader = DataLoader(BraTSDataset(hdf5_path=hgg_hdf5_path, tag='train'), batch_size=batch_size, shuffle=False, num_workers=1) # num_workers MUST be 1: because we use h5py, which cannot be pickled, in torch Dataset.
    # Network
    model = NvNet(inChans=4, input_shape=(160,192,128), seg_outChans=3, activation='relu', normalizaiton='group_normalization', VAE_enable=True, mode='trilinear')

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The number of trainable parametes is {trainable_parameters}.')
    model = model.to(device=device)
    print(model)
    # Loss Function
    criterion = CombinedLoss(k1=0.1, k2=0.1)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    for epoch in range(epoch):

        # Train Model
        print('\n\n\nEpoch: {}\n Train'.format(epoch))
        model.train()
        loss = 0
        learning_rate = learning_rate * (0.5 ** (epoch // 4))
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = learning_rate
        # torch.set_grad_enabled(True)
        for idx, (img, label) in enumerate(dataloader):
            print(img.shape, label.shape)
            
            img, label = img.to(device), label.to(device)
            pred = model(img)
            print(pred.shape)
            exit(0)

            seg_outChans=3
            seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
            batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss)
        log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
        print(log_msg)


        # # Validate Model
        # print('\n\n Validation')
        # net.eval()
        # for module in net.modules():
        #     if isinstance(module, torch.nn.modules.Dropout2d):
        #         module.train(True)
        #     elif isinstance(module, torch.nn.modules.Dropout):
        #         module.train(True)
        #     else:
        #         pass
        # loss = 0
        # torch.set_grad_enabled(False)
        # for idx, (img, label) in enumerate(val_loader):
        #   if torch.cuda.is_available():
        #     img, label = img.cuda(), label.cuda()
        #     pred = net(img)
        #     seg_y_pred, rec_y_pred, y_mid = pred[0][:,:seg_outChans,:,:,:], pred[0][:,seg_outChans:,:,:,:], pred[1]
        #     batch_loss = criterion(seg_y_pred, label, rec_y_pred, img, y_mid)
        #     loss += float(batch_loss)
        # log_msg = '\n'.join(['Epoch: %d  Loss: %.5f' %(epoch, loss/(idx+1))])
        # print(log_msg)

if __name__ == '__main__':
    main()