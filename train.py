from torch.utils.data import DataLoader
import cityscape_dataset
from ssd_net import SSD
from bbox_loss import MultiboxLoss
import torch.optim as optim
import torch
import pandas as pd
import os
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_img_dir_list = pd.read_csv('train_img_dir_list.csv').iloc[:, 1].tolist()
train_json_dir_list = pd.read_csv('train_json_dir_list.csv').iloc[:, 1].tolist()
val_img_dir_list = pd.read_csv('val_img_dir_list.csv').iloc[:, 1].tolist()
val_json_dir_list = pd.read_csv('val_json_dir_list.csv').iloc[:, 1].tolist()
train_dataset = cityscape_dataset.CityScapeDataset(train_img_dir_list, train_json_dir_list, transform = None, mode= "train")
val_dataset = cityscape_dataset.CityScapeDataset(val_img_dir_list, val_json_dir_list, transform = None, mode= "eval")
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, num_workers= 4)

def train(model, optimizer, criterion, num_epochs=5):
    min_total_loss = 1000000
    best_model_wts = copy.deepcopy(model.state_dict())

    output_str = ''
    for idx_epochs in range(num_epochs):
        print("running epoch #", idx_epochs)
        output_str += "running epoch #" + str(idx_epochs) + '\n'
        # train model
        running_conf_loss = 0.0
        running_loc_loss = 0.0
        num_batch = 0
        for batch_idx, (input_img, gt_locs, gt_labels, priors) in enumerate(train_loader):
            print("batch_idx:", batch_idx)
            model.train()
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            # print(input_img.type())
            inputs = input_img.float().to(device)
            # print(inputs.type())
            pred_conf, pred_locs = model.forward(inputs)
            # print(pred_conf)
            gt_labels = gt_labels.to(device)
            gt_locs = gt_locs.to(device)
            conf_loss, loc_loss = criterion.forward(pred_conf, pred_locs, gt_labels, gt_locs)
            print(conf_loss, loc_loss)
            (conf_loss + loc_loss).backward()
            optimizer.step()
            running_conf_loss += conf_loss
            running_loc_loss += loc_loss
            num_batch = batch_idx
            
        
        num_batch += 1
        epoch_conf_loss = running_conf_loss / num_batch
        epoch_loc_loss = running_loc_loss / num_batch
        totall_loss = epoch_conf_loss + epoch_loc_loss

        print('Training Conf Loss: {:.4f} || Loc Loss: {:.4f}'.format(
            epoch_conf_loss, epoch_loc_loss))
        print('Training Totall Loss: {:.4f}'.format(totall_loss))
        output_str += 'Training Conf Loss: {:.4f} || Loc Loss: {:.4f}'.format(
            epoch_conf_loss, epoch_loc_loss) + '\n'
        output_str += 'Training Totall Loss: {:.4f}'.format(totall_loss) + '\n'

        # eval model
        running_conf_loss = 0.0
        running_loc_loss = 0.0
        for batch_idx, (input_img, gt_locs, gt_labels, priors) in enumerate(val_loader):
            model.eval()
            inputs = input_img.float().to(device)
            optimizer.zero_grad()
            torch.set_grad_enabled(False)
            pred_conf, pred_locs = model.forward(inputs)
            gt_labels = gt_labels.to(device)
            gt_locs = gt_locs.to(device)
            conf_loss, loc_loss = criterion.forward(pred_conf, pred_locs, gt_labels, gt_locs)
            running_conf_loss += conf_loss
            running_loc_loss += loc_loss
            num_batch = batch_idx
        num_batch += 1
        epoch_conf_loss = running_conf_loss / num_batch
        epoch_loc_loss = running_loc_loss / num_batch
        totall_loss = epoch_conf_loss + epoch_loc_loss

        print('Eval Conf Loss: {:.4f} || Loc Loss: {:.4f}'.format(
            epoch_conf_loss, epoch_loc_loss))
        print('Eval Totall Loss: {:.4f}'.format(totall_loss))
        output_str += 'Eval Conf Loss: {:.4f} || Loc Loss: {:.4f}'.format(
            epoch_conf_loss, epoch_loc_loss) + '\n'
        output_str += 'Eval Totall Loss: {:.4f}'.format(totall_loss) + '\n' + '-----------' + '\n'
        print('-' * 15)

        with open("Output.txt", "w") as text_file:
            print(output_str, file=text_file)

        if (epoch_conf_loss + epoch_loc_loss) < min_total_loss:
            min_total_loss = epoch_conf_loss + epoch_loc_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(ssd_net, os.path.join(root_dir, 'ssd_net_1022_temporary_best_zoomout.pth'))
            print("saved for temporary best model: ", idx_epochs)
    
    print('Min Val Loss: {:4f}'.format(min_total_loss))

    model.load_state_dict(best_model_wts)
    with open("Output.txt", "w") as text_file:
        print(output_str, file=text_file)
    return model


num_classes = 3
root_dir = 'trained_model'
ssd_net = SSD(num_classes).cuda()
optimizer = optim.Adam(ssd_net.parameters())
criterion = MultiboxLoss()
ssd_net = train(ssd_net, optimizer, criterion, num_epochs = 100)
torch.save(ssd_net, os.path.join(root_dir, 'ssd_net_1023_augmentation_100_zoomout_rgb.pth'))
