import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.re_dataset import re_dataset

from dataset.randaugment import RandomAugment

def create_dataset(config, epoch=None):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    train_dataset = re_dataset(config['train_file'], train_transform, split='train', add_ocr=config['add_ocr'], add_object=config['add_object'])
    val_dataset = re_dataset(config['val_file'], test_transform, split='test', add_ocr=config['add_ocr'], add_object=config['add_object'])       
    test_dataset = re_dataset(config['test_file'], test_transform, split='test', add_ocr=config['add_ocr'], add_object=config['add_object'])

    return train_dataset, val_dataset, test_dataset

def re_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n

def create_loader(datasets, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,bs,n_worker,is_train,collate_fn in zip(datasets,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=None,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
