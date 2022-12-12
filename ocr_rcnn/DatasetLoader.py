import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CollateDataset(object):
    def __call__(self, batch):
        height = [item['img'].shape[1] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        padded_imgs = torch.ones([len(batch), batch[0]['img'].shape[0], max(height), max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                # Padding Image to make them all of equal dimensions
                padded_imgs[idx, :, 0:item['img'].shape[1], 0:item['img'].shape[2]] = item['img']
            except:
                print("Image Padding Failed! Overall Current Shape:")
                print(padded_imgs.shape)
        item = {'idx': indexes, 'img': padded_imgs}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels # Appending list of all labels in current batch
        return item

class LoadDataset(Dataset):
    def __init__(self, param):
        super(LoadDataset, self).__init__()
        self.path = os.path.join(param['local_path'], param['img_dir'])
        self.images = os.listdir(self.path)
        self.total_samples = len(self.images)
        self.image_paths = list(map(lambda x: os.path.join(self.path, x), self.images))
        # Applying following transforms to the image:
        # 1 -> Convert to GrayScale
        # 2 -> Convert to Tensor (Channel, Height, Width) [0.0 - 1.0]
        # 3 -> Normalize (Standardization) the image channel ((input[channel] - mean[channel]) / std[channel])  [-1, 1]
        transform_props =  [transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_props)
        self.collate_fn = CollateDataset()

    # Optional Overwrite
    def __len__(self):
        return self.total_samples

    # Must Overwrite as per PyTorch Docs
    # method to get individual image based on index for the dataset folder
    def __getitem__(self, index):
        if index > len(self):
            raise ValueError("Failed to retrieve image by index!")
        image_path = self.image_paths[index]
        image_file = os.path.basename(image_path)
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        item = {'idx':index, 'img': img}
        item['label'] = image_file.split('_')[0]
        return item 