# версии библиотек и системы, с которыми получилось запустить данный код на локальном компьютере
# Ubuntu 18.04
# python=3.7.13
# torch=1.11.0+cu113
# torchvision=0.12.0

# импорт нужных библиотек для работы с стандартными инструментами python, данными и изображениями
import os
import numpy as np
from pandas.core.indexes.multi import maybe_droplevels
import torch
from PIL import Image
import pandas as pd
import imageio as io
import json
import sys

# здесь прописывается путь до файла transforms.py, который используется далее
sys.path.append('/home/alex/Desktop/Gosha')

import transforms as T

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
            
            # в случае с рамками, подстраивающимися под размер объектов
            #x1 = round(data[i][0] - 50/data[i][2], 2)
            #y1 = round(data[i][1] - 50/data[i][2], 2)
            #x2 = round(data[i][0] + 50/data[i][2], 2)
            #y2 = round(data[i][1] + 50/data[i][2], 2)
            
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

# вспомогательная функция для обработки изображений
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

# создается объект класса с данными
dataset_test = MyData(path_csv_file, path_data, get_transform(train=False))
# номер изображения к которому применяется модель
n = 245
# устройство на котором будет проводиться рассчет
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# изображение к которому применяется
img, _ = dataset_test[n]
# загрузка обученной модели
model = torch.load('/home/alex/Desktop/Gosha/rcnn_model_900_fix_15.pt', map_location=torch.device('cpu'))

# put the model in evaluation mode
model.eval()
# предсказание модели
with torch.no_grad():
  prediction = model([img.to(device)])
# На данном этапе в переменной prediction имеются координаты и метки объектов, предсказанные загруженной моделью model
#----------------------------------------------------------------------------------------------------------------------------------#


# блок ниже визуализирует полученное предсказание
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import cv2
# предсказанные bounding boxes и labels
boxes_pred = prediction[0]['boxes']
labels_pred = prediction[0]['labels'].tolist()
labels_right_pred = ''.join(str(elem) for elem in labels_pred)
# отрисовка предсказанных ответов красным цветом
data = (255 *dataset_test[n][0]).type(torch.uint8)
result = draw_bounding_boxes(data, boxes_pred, labels_right_pred, width = 1, colors='red')

# истинные bounding boxes и labels
boxes = dataset_test[n][1]['boxes']
labels = dataset_test[n][1]['labels'].tolist()
labels_right = ''.join(str(elem) for elem in labels)
# отрисовка истинных ответов синим цветом
result = draw_bounding_boxes(result, boxes, labels_right, width = 1, colors='blue')

#преобразование изображения для корректной отрисовки
plt.imshow(result.permute(1, 2, 0), cmap='hot')
# отрисовка изображения
from torchvision import transforms
im = transforms.ToPILImage()(result).convert("RGB")
plt.show()
