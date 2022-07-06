import os
import numpy as np
from pandas.core.indexes.multi import maybe_droplevels
import torch
from PIL import Image
import pandas as pd
import imageio as io
import json


# класс для обработки данных
class MyData(torch.utils.data.Dataset):
    # метод, вызываемый при инициализации объекта класса, параметры (csv_file - файл с данными, root-место расположения файла с данными)
    def __init__(self, csv_file, root, transforms=None):
        self.root = root
        # функция чтения из csv файла
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
    
    # метод, вызываемый при обращении по индексу к объекту класса
    def __getitem__(self, idx):
        # данный блок извлекает данные из csv файла
        # формат данных в csv файле [индекс картинки, название картинки, массив с координатами центров и размерами объектов, метка класса объекта]
        #                                  [0]              [1]                         [2]                                          [3]
        # конкретные цифры [idx, 1], [idx, 2] зависят от формата данных
        # в моем случае имя картинки находилось под индексом [1],
        # данные об объектах под индексом [2]
        img_name = os.path.join(self.root, self.data.iloc[idx, 1])
        data = json.loads(self.data.iloc[idx, 2])
        boxes = []
        labels = []
        image = Image.open(img_name + '.jpg')
        # имея координаты центров объектов, вычисляются границы bounding boxes
        # в данном случае размер рамок фиксированный и отстоит от центра на 16 пикселей
        for i in range(len(data)):
            y1 = round(data[i][1] - 16, 2)
            x1 = round(data[i][0] - 16, 2)
            x2 = round(data[i][0] + 16, 2)
            y2 = round(data[i][1] + 16, 2)
            boxes.append([x1, y1, x2, y2])
            #добавляются метки класса
            labels.append(data[i][3])
        #производятся преобразования типа данных для обработки алгоритмом
        image_id = torch.tensor([idx])
        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels), ), dtype=torch.int64)

        # в результате имеем данные об ограничиващих рамках, метках объектов и название картинки
        target = {#image_name': self.data.iloc[idx, 1],
        'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd, 'image_name': img_name}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    # метод, вызываемый при вызове метода len()
    def __len__(self):
            return len(self.data)

# путь к файлу с данными
path_csv_file = '/home/alex/Desktop/Gosha/my_data/all_data/data.csv'
# путь к файлу с картинками и данными
path_data = '/home/alex/Desktop/Gosha/my_data/all_data'

my_data = MyData(path_csv_file, path_data)

#----------------------------------------------------------------------------------------------------------------------------#

#  обучение модели
import sys

sys.path.append('drive/MyDrive/Gosha')

from engine import train_one_epoch, evaluate
import utils
import transforms as T

# загрузка предобученной модели
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model




def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    #transforms.append(T.Resize(224))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use our dataset and defined transformations
dataset = MyData(path_csv_file, path_data, get_transform(train=True))
dataset_test = MyData(path_csv_file, path_data, get_transform(train=False))

dataset = dataset
dataset_test = dataset_test

# split the dataset in train and test set
# случайное перемешивание картинок
# 600 изображений в train set 600 - в test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-600])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-600:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has three classes: background, norm-objects, uniform-objects
num_classes = 3

# get the model using our helper function
model = create_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
# SGD оптимизатор с заданными параметрами
optimizer = torch.optim.SGD(params, lr=0.05,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
#один из стандартных lr_scheduler загруженный из torch с заданными параметрами
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10
my_losses = []

# обучение модели
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    losses = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=30)
    losses = list(losses[1:])
    # в my_losses находится значение функции потерь во время обучения
    my_losses += losses
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# сохранение модели с расширением .pt
torch.save(model, 'drive/MyDrive/Gosha/rcnn_model_600_fix_50.pt')