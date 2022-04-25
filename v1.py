import torch, sys
import argparse
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data.dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# import wandb
from dataset import ImageDataset
from training_config import doodles, reals, doodle_size, real_size, NUM_CLASSES
from utils import *  # bad practice, nvm
from losses import compute_contrastive_loss_from_feats

parser = argparse.ArgumentParser(description="WandB model tracking")
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--no-wandb', action='store_false')
# parser.set_defaults(wandb=True)
args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project="cs4243-project", entity="rish-16")

class ExampleMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super(ExampleMLP, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, hid_dim)
        self.l3 = nn.Linear(hid_dim, hid_dim)
        self.l4 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, return_feats=False):
        x = x.flatten(1)  # flatten a pic into a vector
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.l3(x)
        feat = x
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l4(x)

        if return_feats:
            return x, feat

        return x

def train_model(model1, model2, train_set, val_set, tqdm_on,  num_epochs, batch_size, learning_rate, c1, c2, t):
    # cuda side setup
    model1 = nn.DataParallel(model1).cuda()
    model2 = nn.DataParallel(model2).cuda()

    # training side
    optimizer = torch.optim.AdamW(params=list(model1.parameters()) + list(model2.parameters()),
                                  lr=learning_rate, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # load the training data
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16,
                            pin_memory=True, drop_last=True)

    # training loop
    for epoch in range(num_epochs):
        loss1_model1 = AverageMeter()
        loss1_model2 = AverageMeter()
        loss2_model1 = AverageMeter()
        loss2_model2 = AverageMeter()
        loss3_combined = AverageMeter()
        acc_model1 = AverageMeter()
        acc_model2 = AverageMeter()

        model1.train()
        model2.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, y1, x2, y2) in enumerate(pg):
            # doodle, label, real, label
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()

            # train model1 (doodle)
            pred1, feats1 = model1(x1, return_feats=True)
            loss_1 = criterion(pred1, y1)  # classification loss
            loss_2 = compute_contrastive_loss_from_feats(feats1, y1, t)
            loss1_model1.update(loss_1)
            loss2_model1.update(loss_2)
            loss_model1 = loss_1 + c1 * loss_2

            # train model2 (real)
            pred2, feats2 = model2(x2, return_feats=True)
            loss_1 = criterion(pred2, y2)  # classification loss
            loss_2 = compute_contrastive_loss_from_feats(feats2, y2, t)
            loss1_model2.update(loss_1)
            loss2_model2.update(loss_2)
            loss_model2 = loss_1 + c1 * loss_2

            # the third loss
            combined_feat = feats1 * feats2
            loss_3 = compute_contrastive_loss_from_feats(combined_feat, y1, t)
            loss3_combined.update(loss_3)

            loss = loss_model1 + loss_model2 + c2 * loss_3

            # statistics
            acc_model1.update(compute_accuracy(pred1, y1))
            acc_model2.update(compute_accuracy(pred2, y2))

            # optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # display
            pg.set_postfix({
                'acc 1': '{:.6f}'.format(acc_model1.avg),
                'acc 2': '{:.6f}'.format(acc_model2.avg),
                'l1m1': '{:.6f}'.format(loss1_model1.avg),
                'l2m1': '{:.6f}'.format(loss2_model1.avg),
                'l1m2': '{:.6f}'.format(loss1_model2.avg),
                'l2m2': '{:.6f}'.format(loss2_model2.avg),
                'train epoch': '{:03d}'.format(epoch)
            })

        print(
            f'train epoch {epoch}, acc 1={acc_model1.avg:.3f}, acc 2={acc_model2.avg:.3f}, l1m1={loss1_model1.avg:.3f},'
            f'l1m2={loss1_model2.avg:.3f}, l2m1={loss2_model1.avg:.3f}, l2m2={loss2_model2.avg:.3f}, '
            f'l3={loss3_combined.avg:.3f}')

        # validation
        model1.eval(), model1.eval()
        acc_model1.reset(), acc_model2.reset()
        pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)

        with torch.no_grad():
            for i, (x1, y1, x2, y2) in enumerate(pg):
                pred1, feats1 = model1(x1, return_feats=True)
                pred2, feats2 = model2(x2, return_feats=True)
                acc_model1.update(compute_accuracy(pred1, y1))
                acc_model2.update(compute_accuracy(pred2, y2))

                # display
                pg.set_postfix({
                    'acc 1': '{:.6f}'.format(acc_model1.avg),
                    'acc 2': '{:.6f}'.format(acc_model2.avg),
                    'val epoch': '{:03d}'.format(epoch)
                })

                if use_wandb:
                    wandb.log({
                        'acc 1': '{:.6f}'.format(acc_model1.avg),
                        'acc 2': '{:.6f}'.format(acc_model2.avg),
                        'val epoch': '{:03d}'.format(epoch),
                        'acc 1': '{:.6f}'.format(acc_model1.avg),
                        'acc 2': '{:.6f}'.format(acc_model2.avg),
                        'l1m1': '{:.6f}'.format(loss1_model1.avg),
                        'l2m1': '{:.6f}'.format(loss2_model1.avg),
                        'l1m2': '{:.6f}'.format(loss1_model2.avg),
                        'l2m2': '{:.6f}'.format(loss2_model2.avg),
                        'train epoch': '{:03d}'.format(epoch)
                    })

        print(f'validation epoch {epoch}, acc 1 (doodle) = {acc_model1.avg:.3f}, acc 2 (real) = {acc_model2.avg:.3f}')

        scheduler.step()

    print(f'training finished')

    # save checkpoint
    # exp_dir = f'exp_data/{id}'
    # save_model(exp_dir, f'{id}_model1.pt', model1)
    # save_model(exp_dir, f'{id}_model2.pt', model2)


fix_seed(0)  # zero seed by default
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)

# tunable hyper params.
num_epochs, base_bs, base_lr = 20, 512, 2e-2
c1, c2, t = 1, 1, 0.1  # contrastive learning. if you want vanilla (cross-entropy) training, set c1 and c2 to 0.
dropout = 0.2
add_layer = True

# models
doodle_model = ExampleMLP(doodle_size * doodle_size, 128, NUM_CLASSES)
real_model = ExampleMLP(doodle_size * doodle_size, 128, NUM_CLASSES)

config = {
  "learning_rate": base_lr,
  "epochs": num_epochs,
  "batch_size": base_bs
}

if args.wandb:
    wandb.init(config=config)
    config = wandb.config

# just some logistics
tqdm_on = False  # progress bar
# id = 0 # change to the id of each experiment accordingly

train_model(doodle_model, real_model, train_set, val_set, tqdm_on, num_epochs, base_bs, base_lr, c1, c2, t)
