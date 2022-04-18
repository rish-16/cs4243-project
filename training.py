import torch.nn as nn
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses import compute_contrastive_loss_from_feats
from utils import *  # bad practice, nvm
from models import *

ckpt_dir = 'exp_data'


def train_model(model1, model2, train_set, val_set, tqdm_on, id, num_epochs, batch_size, learning_rate, c1, c2, t):
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
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8,
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
            x1 = torch.cat([x1, x1, x1], dim=1)
            pred1, feats1 = model1(x1, return_feats=True)
            loss_1 = criterion(pred1, y1)    # classification loss
            loss_2 = compute_contrastive_loss_from_feats(feats1, y1, t)
            loss1_model1.update(loss_1)
            loss2_model1.update(loss_2)
            loss_model1 = loss_1 + c1 * loss_2

            # train model2 (real)
            pred2, feats2 = model2(x2, return_feats=True)
            loss_1 = criterion(pred2, y2)   # classification loss
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

        print(f'train epoch {epoch}, acc 1={acc_model1.avg:.3f}, acc 2={acc_model2.avg:.3f}, l1m1={loss1_model1.avg:.3f},'
              f'l1m2={loss1_model2.avg:.3f}, l2m1={loss2_model1.avg:.3f}, l2m2={loss2_model2.avg:.3f}, '
              f'l3={loss3_combined.avg:.3f}')

        # validation
        model1.eval(), model1.eval()
        acc_model1.reset(), acc_model2.reset()
        pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
        with torch.no_grad():
            for i, (x1, y1, x2, y2) in enumerate(pg):
                x1 = torch.cat([x1, x1, x1], dim=1)
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

        print(f'validation epoch {epoch}, acc 1 (doodle) = {acc_model1.avg:.3f}, acc 2 (real) = {acc_model2.avg:.3f}')

        scheduler.step()

    print(f'training finished')

    # save checkpoint
    exp_dir = f'exp_data/{id}'
    save_model(exp_dir, f'{id}_model1.pt', model1)
    save_model(exp_dir, f'{id}_model2.pt', model2)


fix_seed(0)         # zero seed by default
if __name__ == "__main__":
    from dataset import ImageDataset
    from training_config import doodles, reals, doodle_size, real_size, NUM_CLASSES

    train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
    val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)

    # tunable hyper params.
    use_cnn = True
    num_epochs, base_bs, base_lr = 20, 256, 1e-2
    c1, c2, t = 1, 1, 0.1  # contrastive learning. if you want vanilla (cross-entropy) training, set c1 and c2 to 0.
    dropout = 0.2

    # models
    doodle_model = ExampleCNN(NUM_CLASSES, dropout) if use_cnn \
        else ExampleMLP(doodle_size * doodle_size, 128, NUM_CLASSES)
    real_model = ExampleCNN(NUM_CLASSES, dropout) if use_cnn \
        else ExampleMLP(doodle_size * doodle_size, 128, NUM_CLASSES)

    # just some logistics
    tqdm_on = False     # progress bar
    id = 0              # change to the id of each experiment accordingly

    train_model(doodle_model, real_model, train_set, val_set, tqdm_on, id, num_epochs, base_bs, base_lr, c1, c2, t)
