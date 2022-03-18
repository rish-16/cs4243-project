import os

import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(ckpt_dir, cp_name, model):
    """
    Create directory /Checkpoint under exp_data_path and save encoder as cp_name
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    saving_model_path = os.path.join(ckpt_dir, cp_name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # convert to non-parallel form
    torch.save(model.state_dict(), saving_model_path)
    print(f'Model saved: {saving_model_path}')


def load_model_dic(model, ckpt_path, verbose=True, strict=True):
    """
    Load weights to model and take care of weight parallelism
    """
    assert os.path.exists(ckpt_path), f"trained model {ckpt_path} does not exist"

    try:
        model.load_state_dict(torch.load(ckpt_path), strict=strict)
    except:
        state_dict = torch.load(ckpt_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=strict)
    if verbose:
        print(f'Model loaded: {ckpt_path}')

    return model
