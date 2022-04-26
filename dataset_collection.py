import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter
import os
import imageio
import glob
from torch.utils.data import Dataset, TensorDataset, DataLoader
import cv2

def sample_array(arr, n=10):
    return arr[np.random.choice(len(arr), n)]

def plot_row(arr):
    """
    plot a row of images in arr
    """
    n = len(arr)
    fig, ax = plt.subplots(1, n)
    plt.gcf().set_size_inches(2*n, 2)
    for i in range(n):
        ax[i].imshow(arr[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()

def plot_dataset(d):
    """
    d is a dict {class: data}
    where class is a string denoting class/label for classification
    and data is a numpy array that is shape (B,H,W)
    
    """
    for clas, data in d.items():
        sample_imgs = sample_array(d[clas])
        plot_row(sample_imgs)
        
def load_dataset(f):
    """
    f is the file name to load a dataset dict of {class: data}
    """
    d = np.load(f, allow_pickle=True)[()]
    assert type(d) == dict
    print(f"Loaded dataset at '{f}'.")
    return d

def save_dataset(f, d):
    assert type(d) == dict
    np.save(f, d)
    print(f"Saved dataset at '{f}'.")
    
def dataset_exists(f):
    return os.path.isfile(f)
    
classes = ['airplane', 'bird', 'car', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']
idx2class = {i: c for i, c in enumerate(classes)}
class2idx = {c: i for i, c in enumerate(classes)}

def get_cifar(f='dataset/cifar'):
    if dataset_exists(f + '/cifar.npy'):
        return load_dataset(f + '/cifar.npy')
    trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)
    
    cifar = {}
    idx_to_class = {v:k for k,v in trainset.class_to_idx.items()}
    for idx, clas in idx_to_class.items():
        idxs = np.where(np.array(trainset.targets) == idx)[0]
        if clas == 'automobile':
            cifar['car'] = trainset.data[idxs]
        elif clas == 'deer':
            continue
        else:
            cifar[clas] = trainset.data[idxs]
    for idx, clas in idx_to_class.items():
        idxs = np.where(np.array(testset.targets) == idx)[0]
        if clas == 'automobile':
            cifar['car'] = np.concatenate([cifar['car'], testset.data[idxs]])
        elif clas == 'deer':
            continue
        else:
            cifar[clas] = np.concatenate([cifar[clas], testset.data[idxs]])
    save_dataset(f"{f}/cifar.npy", cifar)
    return cifar

def print_dataset(d):
    print("Image shape:", d[list(d.keys())[0]].shape[1:])
    print("No. classes:", len(d.keys()))
    print("Classes:", ', '.join(d.keys()))
    print("Count per class:")
    size = 0
    for category, data in d.items():
        print(f"- {category}: {data.shape[0]}")
        size += data.shape[0]
    print("Dataset size:", size)
    
def get_quickdraw(f='dataset/quickdraw'):
    if dataset_exists(f + '/quickdraw.npy'):
        return load_dataset(f + '/quickdraw.npy')
    categories = ['airplane', 'bird', 'car', 'cat', 'dog', 'frog', 'horse', 'cruise ship', 'truck']
    quickdraw = {}
    for c in categories:
        if c == 'cruise ship':
            quickdraw['ship'] = 255 - np.load(f'{f}/{c}.npy').reshape((-1, 28, 28))
        else:
            quickdraw[c] = 255 - np.load(f'{f}/{c}.npy').reshape((-1, 28, 28))
    save_dataset(f"{f}/quickdraw.npy", quickdraw)
    return quickdraw

def get_sketchy_real(f='dataset/sketchy'):
    if dataset_exists(f + '/sketchy_real.npy'):
        return load_dataset(f + '/sketchy_real.npy')
    sketchy_categories = ['airplane', 'songbird', 'wading_bird', 'car_(sedan)', 'cat', 'dog', 'frog', 'horse', 'pickup_truck']
    info = [
        'invalid-ambiguous.txt',
        'invalid-context.txt',
        'invalid-error.txt',
        'invalid-pose.txt']

    remove = []
    for i in info:
        with open(f'{f}/info/{i}', 'r') as file:
            remove += file.read().splitlines()
            
    sketchy_real = {}
    for c in sketchy_categories:
        imgs = []
        for file in glob.glob(f"{f}/photo/tx_000100000000/{c}/*.jpg"):
            name = file[file.index('\\')+1:-4]
            if name in remove:
                print(name)
                continue
            imgs.append(imageio.imread(file))
        if c == 'car_(sedan)':
            sketchy_real['car'] = np.asarray(imgs)
        elif c == 'pickup_truck':
            sketchy_real['truck'] = np.asarray(imgs)
        elif c == 'songbird':
            sketchy_real['bird'] = np.asarray(imgs)
        elif c == 'wading_bird':
            np.append(sketchy_real['bird'], np.asarray(imgs))
        else:
            sketchy_real[c] = np.asarray(imgs)
    save_dataset(f"{f}/sketchy_real.npy", sketchy_real)
    return sketchy_real

def get_sketchy_doodle(f='dataset/sketchy'):
    if dataset_exists(f + '/sketchy_doodle.npy'):
        return load_dataset(f + '/sketchy_doodle.npy')
    sketchy_categories = ['airplane', 'songbird', 'wading_bird', 'car_(sedan)', 'cat', 'dog', 'frog', 'horse', 'pickup_truck']
    info = [
        'invalid-ambiguous.txt',
        'invalid-context.txt',
        'invalid-error.txt',
        'invalid-pose.txt']
    remove = []
    for i in info:
        with open(f'{f}/info/{i}', 'r') as file:
            remove += file.read().splitlines()
    sketchy_doodle = {}
    for c in sketchy_categories:
        imgs = []
        for file in glob.glob(f"{f}/sketch/tx_000000000000/{c}/*.png"):
            name = file[file.index('\\')+1:-4]
            if name in remove:
                continue
            imgs.append(imageio.imread(file))
        if c == 'car_(sedan)':
            sketchy_doodle['car'] = np.asarray(imgs)[:,:,:,0]
        elif c == 'pickup_truck':
            sketchy_doodle['truck'] = np.asarray(imgs)[:,:,:,0]
        elif c == 'songbird':
            sketchy_doodle['bird'] = np.asarray(imgs)[:,:,:,0]
        elif c == 'wading_bird':
            np.append(sketchy_doodle['bird'], np.asarray(imgs)[:,:,:,0])
        else:
            sketchy_doodle[c] = np.asarray(imgs)[:,:,:,0]
    save_dataset(f"{f}/sketchy_doodle.npy", sketchy_doodle)
    return sketchy_doodle

def get_tuberlin(f='dataset/tuberlin'):
    if dataset_exists(f + '/tuberlin.npy'):
        return load_dataset(f + '/tuberlin.npy')
    tuberlin_categories = ['airplane', 'flying bird', 'standing bird', 'car (sedan)', 'race car', 'cat', 'dog', 'frog', 'horse', 'pickup truck', 'truck']
    tuberlin = {}
    for c in tuberlin_categories:
        imgs = []
        for file in glob.glob(f"{f}/{c}/*.png"):
            imgs.append(imageio.imread(file))
        if c == 'car (sedan)':
            tuberlin['car'] = np.asarray(imgs)
        elif c == 'race car':
            np.append(tuberlin['car'], np.asarray(imgs))
        elif c == 'flying bird':
            tuberlin['bird'] = np.asarray(imgs)
        elif c == 'standing bird':
            np.append(tuberlin['bird'], np.asarray(imgs))
        elif c == 'pickup truck':
            tuberlin['truck'] = np.asarray(imgs)
        elif c == 'truck':
            tuberlin['truck'] = np.asarray(imgs)
            np.append(tuberlin['truck'], np.asarray(imgs))
        else:
            tuberlin[c] = np.asarray(imgs)
    save_dataset(f"{f}/tuberlin.npy", tuberlin)
    return tuberlin

def get_google_doodles(f='dataset/google_images'):
    return load_dataset(f'{f}/google_doodles.npy')

def get_google_real(f='dataset/google_images'):
    return load_dataset(f'{f}/google_real.npy')

def train_test_split(d, split=0.8, shuffle=True):
    train_set = {}
    test_set = {}
    for clas, data in d.items():
        if shuffle:
            np.random.shuffle(data)
        n = data.shape[0]
        train, test = data[:int(n*split)], data[int(n*split):]
        train_set[clas] = train
        test_set[clas] = test
    return train_set, test_set


def get_all_datasets():
    dd = {
        'cifar': get_cifar(),
        'quickdraw': get_quickdraw(),
        'sketchy_real': get_sketchy_real(),
        'sketchy_doodle': get_sketchy_doodle(),
        'tuberlin': get_tuberlin(),
        'google_doodles': get_google_doodles(),
        'google_real': get_google_real()}
    return dd

def get_doodle_datasets():
    dd = {
        'quickdraw': get_quickdraw(),
        'sketchy_doodle': get_sketchy_doodle(),
        'tuberlin': get_tuberlin(),
        'google_doodles': get_google_doodles()}
    return dd

def get_real_datasets():
    dd = {
        'cifar': get_cifar(),
        'sketchy_real': get_sketchy_real(),
        'google_real': get_google_real()}
    return dd

def collapse_datasets(dd, res=64):
    """
    Collapses all datasets in dd, a dict of dicts,
    and resizes images to the same specified resolution.
    """
    cd = {}
    for name, d in dd.items():
        for c, data in d.items():
            if c not in cd:
                cd[c] = []
            resized = []
            for img in data:
                resized.append(cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA))
            resized = np.stack(resized, axis=0)
            cd[c].append(resized)
    for c, lst_data in cd.items():
        cd[c] = np.concatenate(lst_data, axis=0)
    return cd
              
              
def get_sketchy_pairs(f='dataset/sketchy'):
    if dataset_exists(f + '/sketchy_pairs.npy'):
        return load_dataset(f + '/sketchy_pairs.npy')
    sketchy2class = {
        'airplane': 'airplane',
        'songbird': 'bird',
        'wading_bird': 'bird',
        'car_(sedan)': 'car',
        'cat': 'cat',
        'dog': 'dog',
        'frog': 'frog',
        'horse': 'horse',
        'pickup_truck': 'truck',
    }
    def check_real(url):
        assert url[-4:] == '.jpg'
        assert '-' not in url
    def check_doodle(url):
        assert url[-4:] == '.png'
        assert '-' in url
    sketchy_categories = ['airplane', 'songbird', 'wading_bird', 'car_(sedan)', 'cat', 'dog', 'frog', 'horse', 'pickup_truck']
    info = [
        'invalid-ambiguous.txt',
        'invalid-context.txt',
        'invalid-error.txt',
        'invalid-pose.txt']
    remove = []
    for i in info:
        with open(f'{f}/info/{i}', 'r') as file:
            remove += file.read().splitlines()
    sketchy_pairs = {c: {} for c in classes}
    for c in sketchy_categories:
        for file in glob.glob(f"{f}/photo/tx_000100000000/{c}/*.jpg"):
            check_real(file)
            name = file[file.index('\\')+1:-4]
            img = np.asarray(imageio.imread(file))
            if name not in sketchy_pairs[sketchy2class[c]]:
                sketchy_pairs[sketchy2class[c]][name] = {'real': img, 'doodle': []}
            else:
                raise Exception(f"{name} exists")

        for file in glob.glob(f"{f}/sketch/tx_000000000000/{c}/*.png"):
            check_doodle(file)
            name = file[file.index('\\')+1:file.index('-')]
            idx = file[file.index('-')+1:-4]
            img = np.asarray(imageio.imread(file))
            if name in sketchy_pairs[sketchy2class[c]]:
                sketchy_pairs[sketchy2class[c]][name]['doodle'].append(img[:,:,0])
            else:
                raise Exception(f"{name} does not have a real pair")
    doodles = []
    reals = []
    labels = []
    idxs = []
    for c, idx in sketchy_pairs.items():
        for name, data in idx.items():
            for d in data['doodle']:
                doodles.append(d)
                reals.append(data['real'])
                labels.append(class2idx[c])
                idxs.append(name)
    doodles = np.asarray(doodles)
    reals = np.asarray(reals)
    labels = np.asarray(classes)
    idxs = np.asarray(idxs)
    sketchy_pairs = {
        'idxs': idxs,
        'doodles': doodles,
        'reals': reals,
        'labels': labels}
    save_dataset(f"{f}/sketchy_pairs.npy", sketchy_pairs)
    return sketchy_pairs