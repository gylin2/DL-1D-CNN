import argparse
import os
import json
import shutil
from utils import *
from dataset import VPCID
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import torchaudio
from model import Model
import sys
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

torch.set_num_threads(4)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Data folder prepare               
    parser.add_argument("-p", "--path", type=str, help="VPCID path",
                        default='/dataset_paths')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models/TEMP')
    parser.add_argument('--case', type=str, default="case0",help="Type of dataset case")

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=25, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")

    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")

    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--seed', type=int, help="random number seed", default=0)
    
    parser.add_argument('--num_blocks', type=int, default=2, help="number of workers")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    

    # Set seeds
    setup_seed(args.seed)
    
    # Path for output data
    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    else:
        shutil.rmtree(args.out_fold)
        os.mkdir(args.out_fold)

    # Save training arguments
    with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

    # assign device
    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def train(args):
    torch.set_default_dtype(torch.float32)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    model = Model(args)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=20, log_mels=True,
                                melkwargs={"n_fft": 80, "hop_length": 40, "n_mels": 32, "center": False},).to(args.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(args.beta_1, args.beta_2), eps=args.eps, 
                                    weight_decay=0, amsgrad=True)
    
    training_set = VPCID(args.path, 'train', args.case)
    validation_set = VPCID(args.path, 'val', args.case)
    testing_set = VPCID(args.path, 'test', args.case)                 
    trainDataLoader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True,
                                 )
    valDataLoader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                               )
    testDataLoader = DataLoader(testing_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                               )
    if args.case == 'case61':
        case62_set = VPCID(args.path, 'test', 'case62')
        case62DataLoader = DataLoader(case62_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                               )
        
    feat, _, _ = training_set[29]
    print("Feature shape", feat.shape)

    criterion = nn.CrossEntropyLoss().to(args.device)
    prev_acc = 0

    for epoch_num in tqdm(range(args.num_epochs)):
        model.train()
        devlossDict = defaultdict(list)
        logging.info('\nEpoch: %d ' % (epoch_num + 1))
        idx_loader, score_loader = [], []

        if sys.stdout.isatty():
            training_loader = tqdm(trainDataLoader)
        else:
            training_loader = trainDataLoader 

        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(args.device).float()+ 1e-12
            labels = labels.to(args.device, non_blocking=True).long()
            inputs = mfcc_transform(inputs)
            _, conv_out, output = model(inputs)
            optimizer.zero_grad()

            main_loss = criterion(output, labels)
            frames_prob = F.log_softmax(conv_out, dim=1)
            frames_prob = torch.mean(frames_prob,  dim=-1)
            frames_loss = F.nll_loss(frames_prob, labels)

            main_loss = main_loss + frames_loss
            main_loss.backward()
            optimizer.step()
        

        # Val the model
        model.eval()
        with torch.no_grad():
            if sys.stdout.isatty():
                val_loader = tqdm(valDataLoader)
            else:
                val_loader = valDataLoader  
            idx_loader, score_loader = [], []
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.float().to(args.device, non_blocking=True)+ 1e-12
                labels = labels.to(args.device, non_blocking=True).long()
                inputs = mfcc_transform(inputs)
                _, conv_out, output = model(inputs)

                main_loss = criterion(output, labels)
                frames_prob = F.log_softmax(conv_out, dim=1)
                frames_prob = torch.mean(frames_prob,  dim=-1)
                frames_loss = F.nll_loss(frames_prob, labels)

                conv_out = torch.mean(F.softmax(conv_out, dim=1), dim=-1)

                devlossDict['frames_loss'].append(frames_loss.item())
                devlossDict['main_loss'].append(main_loss.item())
                idx_loader.append(labels)
                score_loader.append(output)


            scores = torch.cat(score_loader, 0) #.data.cpu().numpy()
            labels = torch.cat(idx_loader, 0)  #.data.cpu().numpy()
            val_acc = (scores.argmax(1) == labels).float().mean()

            avg_frames_loss = np.nanmean(devlossDict['frames_loss'])
            avg_main_loss = np.nanmean(devlossDict['main_loss'])
        
        save_checkpoint(model, optimizer, args.lr, epoch_num, os.path.join(args.out_fold, 'model_'+str(epoch_num)+'.pt'))
        if epoch_num == args.num_epochs-1:
            save_checkpoint(model, optimizer, args.lr, epoch_num, os.path.join(args.out_fold, 'last_model.pt'))
        if val_acc >= prev_acc:
            # Save the model checkpoint
            save_checkpoint(model, optimizer, args.lr, epoch_num, os.path.join(args.out_fold, 'best_model.pt'))
            prev_acc = val_acc
        logging.info('main_loss: {:.4f} - frames_loss: {:.4f} - acc: {:.4f} - best_acc: {:.4f}'.format\
                        (avg_main_loss, avg_frames_loss, val_acc, prev_acc))
        
        if prev_acc == val_acc:
            model.eval()
            with torch.no_grad():
                if sys.stdout.isatty():
                    test_loader = tqdm(testDataLoader)
                else:
                    test_loader = testDataLoader 
                idx_loader, score_loader = [], []
                frames_score_loader = []
                for i, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.float().to(args.device, non_blocking=True)+ 1e-12
                    labels = labels.to(args.device, non_blocking=True).long()
                    inputs = mfcc_transform(inputs)
                    _, conv_out, outputs = model(inputs)
                    conv_out = torch.mean(F.softmax(conv_out, dim=1), dim=-1)

                    idx_loader.append(labels)
                    score_loader.append(outputs.squeeze())
                    frames_score_loader.append(conv_out.squeeze())
                scores = torch.cat(score_loader, 0) #.data.cpu().numpy()
                frames_score = torch.cat(frames_score_loader, 0) #.data.cpu().numpy()
                labels = torch.cat(idx_loader, 0)  #.data.cpu().numpy()
                TPR, TNR, F1, AUC, ACC = calculate_metric(scores, labels)
                _, _, _, _, ACC_frames = calculate_metric(frames_score, labels)
                logging.info('TPR: {:.6f} - TNR: {:.6f} - F1: {:.6f} - AUC: {:.6f} - ACC: {:.6f} - ACC_f: {:.6f}'.format(TPR, TNR, F1, AUC, ACC, ACC_frames))
            
            if args.case == 'case61':
                model.eval()
                with torch.no_grad():
                    if sys.stdout.isatty():
                        test_loader = tqdm(case62DataLoader)
                    else:
                        test_loader = case62DataLoader 
                    idx_loader, score_loader = [], []
                    frames_score_loader = []
                    for i, (inputs, inputs2, labels) in enumerate(test_loader):
                        inputs = inputs.float().to(args.device, non_blocking=True)+ 1e-12
                        labels = labels.to(args.device, non_blocking=True).long()
                        inputs = mfcc_transform(inputs)
                        _, conv_out, outputs = model(inputs)
                        conv_out = torch.mean(F.softmax(conv_out, dim=1), dim=-1)

                        idx_loader.append(labels)
                        score_loader.append(outputs.squeeze())
                        frames_score_loader.append(conv_out.squeeze())
                    scores = torch.cat(score_loader, 0) #.data.cpu().numpy()
                    frames_score = torch.cat(frames_score_loader, 0) #.data.cpu().numpy()
                    labels = torch.cat(idx_loader, 0)  #.data.cpu().numpy()
                    c62_TPR, c62_TNR, c62_F1, c62_AUC, c62_ACC = calculate_metric(scores, labels)
                    _, _, _, _, c62_ACC_frames = calculate_metric(frames_score, labels)
                    logging.info('TPR: {:.6f} - TNR: {:.6f} - F1: {:.6f} - AUC: {:.6f} - ACC: {:.6f} - ACC_f: {:.6f}'.format(
                        c62_TPR, c62_TNR, c62_F1, c62_AUC, c62_ACC, c62_ACC_frames))
                    
        if epoch_num == args.num_epochs-1:
            logging.info('TPR: {:.6f} - TNR: {:.6f} - F1: {:.6f} - AUC: {:.6f} - ACC: {:.6f} - ACC_f: {:.6f}'.format(TPR, TNR, F1, AUC, ACC, ACC_frames))
            if args.case == 'case61':
                logging.info('TPR: {:.6f} - TNR: {:.6f} - F1: {:.6f} - AUC: {:.6f} - ACC: {:.6f} - ACC_f: {:.6f}'.format(
                    c62_TPR, c62_TNR, c62_F1, c62_AUC, c62_ACC, c62_ACC_frames))
    return model


if __name__ == "__main__":
    args = initParams()
    train(args)
