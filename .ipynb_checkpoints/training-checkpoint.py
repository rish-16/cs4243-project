import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TrainSamplerMultiClassUnit
from losses import compute_sim_matrix, compute_target_matrix, contrastive_loss
from utils import *  # bad practice, nvm

ckpt_dir = 'exp_data'


def train_bert(train_set, val_set, model, tqdm_on, id, num_epochs, base_bs, base_lr,
               coefficient, num_authors):
    ngpus, dropout = torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 512, 512, num_authors
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    temperature, sample_unit_size = 0.1, 2
    print(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')

    # logger
    exp_dir = os.path.join(ckpt_dir,
                           f'{id}_coe{coefficient}_temp{temperature}_unit{sample_unit_size}_epoch{num_epochs}')
    writer = SummaryWriter(os.path.join(exp_dir, 'board'))

    # load the training data
    train_sampler = TrainSamplerMultiClassUnit(train_set, sample_unit_size=sample_unit_size)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, sampler=train_sampler, shuffle=False,
                              num_workers=4 * ngpus, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                            pin_memory=True, drop_last=True)

    # training loop
    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()

        # decay coefficient
        # coefficient = coefficient - 1 / num_epochs

        model.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, x2, x3, y) in enumerate(pg):  # for x1, x2, x3, y in train_set:
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            # classification loss
            loss_1 = criterion(pred, y.long())

            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            target_matrix = compute_target_matrix(y)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

            # total loss
            loss = loss_1 + coefficient * loss_2

            acc = (pred.argmax(1) == y).sum().item() / len(y)
            train_acc.update(acc)
            train_loss.update(loss.item())
            train_loss_1.update(loss_1.item())
            train_loss_2.update(loss_2.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train L1': '{:.6f}'.format(train_loss_1.avg),
                'train L2': '{:.6f}'.format(train_loss_2.avg),
                'train L': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

            # iteration logger
            step = i + epoch * len(pg)
            writer.add_scalar("train-iteration/L1", loss_1.item(), step)
            writer.add_scalar("train-iteration/L2", loss_2.item(), step)
            writer.add_scalar("train-iteration/L", loss.item(), step)
            writer.add_scalar("train-iteration/acc", acc, step)

        print('train acc: {:.6f}'.format(train_acc.avg), 'train L1 {:.6f}'.format(train_loss_1.avg),
              'train L2 {:.6f}'.format(train_loss_2.avg), 'train L {:.6f}'.format(train_loss.avg), f'epoch {epoch}')

        # epoch logger
        writer.add_scalar("train/L1", train_loss_1.avg, epoch)
        writer.add_scalar("train/L2", train_loss_2.avg, epoch)
        writer.add_scalar("train/L", train_loss.avg, epoch)
        writer.add_scalar("train/acc", train_acc.avg, epoch)

        # train_val
        model.eval()
        pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
        val_acc = AverageMeter()  # tv stands for train_val
        val_loss_1 = AverageMeter()
        val_loss_2 = AverageMeter()
        val_loss = AverageMeter()
        with torch.no_grad():
            for i, (x1, x2, x3, y) in enumerate(pg):
                x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                pred, feats = model(x, return_feat=True)

                # classification
                loss_1 = criterion(pred, y.long())

                # contrastive learning
                sim_matrix = compute_sim_matrix(feats)
                target_matrix = compute_target_matrix(y)
                loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                # total loss
                loss = loss_1 + coefficient * loss_2

                # logger
                val_acc.update((pred.argmax(1) == y).sum().item() / len(y))
                val_loss.update(loss.item())
                val_loss_1.update(loss_1.item())
                val_loss_2.update(loss_2.item())

                pg.set_postfix({
                    'train_val acc': '{:.6f}'.format(val_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # scheduler.step(test_loss.avg)
        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {val_acc.avg}, val acc {val_acc.avg}, '
              f'val loss {val_loss}')

    # save checkpoint
    save_model(exp_dir, f'{id}_val{val_acc:.5f}_finale{epoch}.pt', model)
