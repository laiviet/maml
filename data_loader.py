import torch
import pickle
import random
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import ToPILImage, ToTensor, Resize

W, H, C = 84, 84, 3


class FSLDataset(Dataset):

    def __init__(self, part='train', iteration=500, fsl_setting=(4, 5, 5, 4)):
        super(FSLDataset, self).__init__()
        assert part in ['train', 'val', 'test']
        self.B, self.N, self.K, self.Q = fsl_setting
        self.length = iteration * fsl_setting[0]

        file = open('data/mini-imagenet-cache-{}.pkl'.format(part), "rb")
        print('Loadding: ', file)
        data = pickle.load(file)
        self.data = data['image_data']
        self.class_dict = data['class_dict']
        self.n_class = self.data.shape[0] / 600
        self.list_of_class = list(self.class_dict.keys())

        self.transform = torchvision.transforms.Compose(
            [ToPILImage(),
             Resize([28, 28]),
             ToTensor()]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        N, K, Q = self.N, self.K, self.Q
        classes = random.sample(self.list_of_class, N)
        support_targets, query_targets = [], []
        support_indices, query_indices = [], []
        for i, c in enumerate(classes):
            indices = random.sample(self.class_dict[c], K + Q)
            # print(indices)
            support_indices += indices[:K]
            query_indices += indices[K:]
            support_targets += [i for _ in range(K)]
            query_targets += [i for _ in range(Q)]
        support_targets = torch.LongTensor(support_targets)
        query_targets = torch.LongTensor(query_targets)

        # print(support_indices)

        support_images = []
        for index in support_indices:
            image = self.data[index]
            # print('Image shape', image.shape)
            image = self.transform(image)
            support_images.append(image)
        support_images = torch.stack(support_images)

        query_images = []
        for index in query_indices:
            image = self.data[index]

            # print('Image shape', image.shape)
            image = self.transform(image)
            query_images.append(image)
        query_images = torch.stack(query_images)

        return support_images, query_images, support_targets, query_targets


if __name__ == '__main__':
    d = FSLDataset('val')
    d = DataLoader(d, batch_size=4)
    for i, (batch, labels) in enumerate(d):
        if i == 0:
            print(type(batch), batch.shape)
            print(type(labels), labels.shape)
        print(i)
