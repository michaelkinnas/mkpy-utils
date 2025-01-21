from numpy import  array
from PIL import Image
from os import listdir, path, walk
from random import Random
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, root: str, transform = None, target_transform = None, task = 'classification', split = 'train', seed=None):
        '''
        Currently only classification task is supported.
        '''            
        self.transform = transform
        self.target_transform = target_transform
        self.random = Random(seed)

        images = []
        for i, classdir in enumerate(listdir(path.join(root, "train"))):
            for image in listdir(path.join(root, "train", classdir, "images")):
                img = Image.open(path.join(root, "train", classdir, "images", image))
                if img.mode not in ['RGB']:
                    img = img.convert('RGB')
                images.append((array(img), i))

        # shuffle images
        self.random.shuffle(images)
        self.data = [x[0] for x in images]
        self.targets = [x[1] for x in images]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class INTEL(Dataset):
    def __init__(self, root, transform=None, target_transform = None, split: str = 'train', seed=None):
        self.directory = "intel"
        self.random = Random(seed)
        
        self.transform = transform
        self.target_transform = target_transform

        self.labels = {
            'buildings' : 0,
            'forest': 1, 
            'glacier' : 2, 
            'mountain' : 3, 
            'sea' : 4, 
            'street' : 5,
        }

        if split not in ['train', 'test']:
            raise "Wrong split specified. Must be either train or test"

        images = []
        for label in self.labels.keys():
            filepath = path.join(*[root, self.directory, 'seg_'+split+'/seg_'+split, label])
            image_files = list(walk(filepath))[0][2]
            for image_file in image_files:
                image_file_path = path.join(filepath, image_file)
                image = Image.open(image_file_path)
                images.append((array(image), array(self.labels[label])))

        self.random.shuffle(images)        
        self.data = [x[0] for x in images]
        self.targets = [x[1] for x in images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):      
        img, target = self.data[idx], self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)        

        if self.target_transform is not None:
            target = self.target_transform(target)
       
        return img, target