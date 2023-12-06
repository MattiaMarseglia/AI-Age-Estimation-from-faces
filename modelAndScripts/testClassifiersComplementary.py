#This code is to try a single regressor with our backbone
import os
import cv2
import dlib
import json
import copy
import torch
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchscan import summary #network summary
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import train_test_split
from torchvision.models import Inception_V3_Weights, Inception3



PARAMETERS_AND_NAME_MODEL = 'totalModel'


print(torch.cuda.device_count())

CUDA_VISIBLE_DEVICES=0
torch.cuda.set_per_process_memory_fraction(0.16, CUDA_VISIBLE_DEVICES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
        

class dataFrameDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        age = self.df.iloc[idx,1]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'age': age}
        return sample

def preprocess_input(img):
    return  img

class allignment(object):
  
  def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.
        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

        self.randomBrightness = transforms.ColorJitter(brightness=(0.9,1.1))



  def shape_to_normal(self, shape):
      shape_normal = []
      for i in range(0, 5):
          shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
      return shape_normal
  def get_eyes_nose_dlib(self, shape):
      nose = shape[4][1]
      left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
      left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
      right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
      right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
      return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)
  def cosine_formula(self, length_line1, length_line2, length_line3):
      cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
      return cos_a
  def distance(self, a, b):
      return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
  def rotate_point(self, origin, point, angle):
      ox, oy = origin
      px, py = point

      qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
      qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
      return qx, qy


  def is_between(self, point1, point2, point3, extra_point):
      c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
      c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
      c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
      if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
          return True
      else:
          return False

  def load_image(self, path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

  def allignImage(self, img):       
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rects = self.detector(gray, 0)
    angle = 0
    try:
        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()
                shape = self.predictor(gray, rect)
            shape = self.shape_to_normal(shape)
            nose, left_eye, right_eye = self.get_eyes_nose_dlib(shape)
            center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            center_pred = (int((x + w) / 2), int((y + y) / 2))
            length_line1 = self.distance(center_of_forehead, nose)
            length_line2 = self.distance(center_pred, nose)
            length_line3 = self.distance(center_pred, center_of_forehead)
            cos_a = self.cosine_formula(length_line1, length_line2, length_line3)
            try:
                angle = np.arccos(cos_a)
            except:
                angle = 0
            rotated_point = self.rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
            if self.is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)
    except:
        pass
    img = Image.fromarray(img)
    return np.array(img.rotate(angle))
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches
modelPath = ('./landmarks5.dat')

allignment = allignment(modelPath)

class face_alignment_train(object):

    def __call__(self,img):
        
        if bool(random.getrandbits(1)): 
            img = allignment.randomBrightness(img)

        img = np.array(img) 
        # Convert RGB to BGR 
        img = allignment.allignImage(img).astype(np.uint8)

        if random.random() > 0.7:
            img = cv2.flip(img, 1)

        
        angle = 0 
        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype(np.uint8))

        if bool(random.getrandbits(1)): 
            if bool(random.getrandbits(1)): 
                angle = 5
            else:
                angle = -5
        return preprocess_input(img.rotate(angle))

class face_alignment_val(object):

    def __call__(self,img): 
        
        img = np.array(img) 

        img = allignment.allignImage(img).astype(np.uint8)
        
        return preprocess_input(img)

df = pd.read_csv(
   '/user/2022_va_gr12/training_caip_contest.csv',
    names=["image", "age"],dtype={'image':'str','age':'float'})
df.head()


X_train , X_val, y_train, y_val = train_test_split(df['image'],df['age'],train_size=0.74, shuffle=True,stratify=df['age'])



df_train = pd.DataFrame({'image':X_train,'age':y_train})
df_val = pd.DataFrame({'image':X_val,'age':y_val})


import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

data_transforms_train = transforms.Compose([
transforms.Resize((160,160)), 
face_alignment_train(),
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

data_transforms_val = transforms.Compose([
transforms.Resize((160,160)), 
face_alignment_val(),
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

PATH = "/mnt/sdc1/2022_va_gr12/training_caip_contest"

trainDataSet = dataFrameDataset(df_train,PATH,data_transforms_train)
valnDataSet = dataFrameDataset(df_val,PATH,data_transforms_val)
batch_size = 256
# create batches

train_set = DataLoader(trainDataSet, shuffle=True,batch_size=batch_size,num_workers=15)
val_set = DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=15)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}

class eoClassifierModel(nn.Module):
    def __init__(self, Backbone, LasteoClassifierModel):
        super(eoClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.LasteoClassifierModel = LasteoClassifierModel
        
    def forward(self, image):
        featureVect = self.Backbone(image)
        output = self.LasteoClassifierModel(featureVect)
        return output

#DEFINE MODEL

#BACKBONE:
from facenet_pytorch import InceptionResnetV1

# For a model pretrained on VGGFace2
backbone1 = InceptionResnetV1(pretrained='vggface2')
backbone2 = InceptionResnetV1(pretrained='vggface2')

num_ftrs = backbone1.last_linear.out_features

#CLASSIFIER
ClassifierModel1 = nn.Sequential(
    nn.Linear(num_ftrs, num_ftrs),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(num_ftrs, int(num_ftrs/2)),
    nn.BatchNorm1d(num_features = int(num_ftrs/2)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/2), int(num_ftrs/4)),
    nn.BatchNorm1d(int(num_ftrs/4)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/4), 81),
)

ClassifierModel2 = nn.Sequential(
    nn.Linear(num_ftrs, num_ftrs),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(num_ftrs, int(num_ftrs/2)),
    nn.BatchNorm1d(num_features = int(num_ftrs/2)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/2), int(num_ftrs/4)),
    nn.BatchNorm1d(int(num_ftrs/4)),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(int(num_ftrs/4), 81),
)

##BACKBONE HEAD FOR REGRESSOR
RegressorHead=copy.deepcopy(nn.Sequential(
        backbone1.mixed_7a,
        backbone1.repeat_3,
        backbone1.block8,
        backbone1.avgpool_1a,
        backbone1.dropout,
        backbone1.last_linear,
        backbone1.last_bn
    ))

#CLASSIFIER MODEL
classifier1 = eoClassifierModel(backbone1, ClassifierModel1)
classifier2 = eoClassifierModel(backbone2, ClassifierModel2)

checkpoint1 = torch.load('./BoostingSeed/primoClassificatoreArgMax_updated.pt')
checkpoint1.keys()
classifier1.load_state_dict(checkpoint1)

checkpoint2 = torch.load('./BoostingSeed/secondoClassificatoreArgMax_updated.pt')
checkpoint2.keys()
classifier2.load_state_dict(checkpoint2)

count = 0
for param in classifier1.parameters(): 
    param.requires_grad = False
    count+=1
print(count)

count = 0
for param in classifier2.parameters(): 
    param.requires_grad = False
    count+=1
print(count)

import collections


#####EVAL PRIMO CLASSIFIFCATORE#####
def adaboost_clf(train_set, model1, model2, rangeNum):
    miss1 = []
    miss2 = []
    orMiss = []
    andMiss = []
    possibleChoise = torch.arange(1,82).unsqueeze(0).to(device)
    # possibleChoise = torch.arange(1,82).unsqueeze(0).to(device)
    for sample_batched in tqdm(train_set):
        inputs = sample_batched['image'].float().to(device)
        labels = sample_batched['age'].float().to(device)

        pred_train_i_1 = model1(inputs)
        pred_train_i_2 = model2(inputs)

        prob1 = nn.Softmax(dim=1)(pred_train_i_1)
        prob2 = nn.Softmax(dim=1)(pred_train_i_2)

        preds1 = torch.argmax(prob1,dim=1)
        preds2 = torch.argmax(prob2,dim=1)

        preds1 += 1
        preds2 += 1

        # Indicator function
        miss1 = np.concatenate((miss1, [int(x) for x in (torch.logical_and(preds1 >= labels.data-rangeNum, preds1 <= labels.data+rangeNum)).long()]))
        miss2 = np.concatenate((miss2, [int(x) for x in (torch.logical_and(preds2 >= labels.data-rangeNum, preds2 <= labels.data+rangeNum)).long()]))
    
    andMiss = np.logical_and(miss1, miss2)
    orMiss = np.logical_or(miss1, miss2)

    # Equivalent with 1/-1 to update weights
    andValues = collections.Counter(andMiss)
    orValues = collections.Counter(orMiss)

    Sum = andValues[0] + andValues[1]
    andAccuracy = andValues[1]/Sum
    orAccuracy = orValues[1]/Sum
    return andAccuracy, orAccuracy


classifier1 = classifier1.to(device)
classifier2 = classifier2.to(device)

classifier1.eval()
classifier2.eval()

summary(classifier1, (3,160,160))
summary(classifier2, (3,160,160))

train_set = DataLoader(trainDataSet,shuffle=False,batch_size=256,num_workers=15)
val_set = DataLoader(valnDataSet,shuffle=False, batch_size=256,num_workers=15)


andTrain1,orTrain1 = adaboost_clf(train_set, classifier1, classifier2, 1)
andVal1,orVal1 = adaboost_clf(val_set, classifier1, classifier2, 1)
print("rangeNum", 1, "; accuracy or train:", orTrain1*100, "accuracy or val:", orVal1*100, "accuracy and train:", andTrain1*100, "accuracy and val:", andVal1*100)


andTrain1,orTrain1 = adaboost_clf(train_set, classifier1, classifier2, 2)
andVal1,orVal1 = adaboost_clf(val_set, classifier1, classifier2, 2)
print("rangeNum", 2, "; accuracy or train:", orTrain1*100, "accuracy or val:", orVal1*100, "accuracy and train:", andTrain1*100, "accuracy and val:", andVal1*100)

andTrain1,orTrain1 = adaboost_clf(train_set, classifier1, classifier2, 3)
andVal1,orVal1 = adaboost_clf(val_set, classifier1, classifier2, 3)
print("rangeNum", 3, "; accuracy or train:", orTrain1*100, "accuracy or val:", orVal1*100, "accuracy and train:", andTrain1*100, "accuracy and val:", andVal1*100)

andTrain1,orTrain1 = adaboost_clf(train_set, classifier1, classifier2, 4)
andVal1,orVal1 = adaboost_clf(val_set, classifier1, classifier2, 4)
print("rangeNum", 4, "; accuracy or train:", orTrain1*100, "accuracy or val:", orVal1*100, "accuracy and train:", andTrain1*100, "accuracy and val:", andVal1*100)


#VECCHI CLASSIFICATORI
# nohup python3 ./Boosting/testClassifiersComplementary.py > ./Boosting/testClassifiersComplementaryargMaxCCE.txt &

                        # Classifier 2 argMax
# rangeNum 1 ; accuracy or train: 54.30709146195313 accuracy or val: 55.103364789759155 accuracy and train: 8.381779985618747 accuracy and val: 8.56212253961035
# rangeNum 2 ; accuracy or train: 74.26061087429562 accuracy or val: 75.10751141995333 accuracy and train: 21.892403784243598 accuracy and val: 22.507507407085388
# rangeNum 3 ; accuracy or train: 85.4547248997777 accuracy or val: 86.23853824597543 accuracy and train: 36.248278714334724 accuracy and val: 37.12772289809322
# rangeNum 4 ; accuracy or train: 91.58085695352412 accuracy or val: 92.13210361224995 accuracy and train: 49.351903636201286 accuracy and val: 50.20499067008206

                        # Classifier argMax e EV
# rangeNum 1 ; accuracy or train: 56.497177796472364 accuracy or val: 57.66624977427618 accuracy and train: 11.527561719546755 accuracy and val: 11.912198449695357
# rangeNum 2 ; accuracy or train: 77.06589528003497 accuracy or val: 78.27834589583932 accuracy and train: 33.47753751580293 accuracy and val: 34.528053290886106
# rangeNum 3 ; accuracy or train: 87.00893423631314 accuracy or val: 87.85037353112314 accuracy and train: 54.487562095527245 accuracy and val: 55.748098903818246
# rangeNum 4 ; accuracy or train: 92.3713559266274 accuracy or val: 92.89655495288224 accuracy and train: 69.81769646155365 accuracy and val: 71.04046977307232

#NUOVI CLASSIFICATORI
                        # Classifier argMax e EV
#rangeNum 1 ; accuracy or train: 70.0804128265743 accuracy or val: 73.46825487061845 accuracy and train: 30.300267416121102 accuracy and val: 32.678121175235255
#rangeNum 2 ; accuracy or train: 86.91329420003102 accuracy or val: 89.2996876651128 accuracy and train: 64.34765035694647 accuracy and val: 67.78937793858975
#rangeNum 3 ; accuracy or train: 93.9666881288861 accuracy or val: 95.35577418254536 accuracy and train: 83.13821512663493 accuracy and val: 85.95094937767107
#rangeNum 4 ; accuracy or train: 96.976176936417 accuracy or val: 97.65916037426682 accuracy and train: 91.86401725750434 accuracy and val: 93.47641436874244

#SUPER NUOVI CLASSIFICATORI (argMax)
                        # Classifier argMax e EV
# (anche singolarmente più alti, guardate il val)
#rangeNum 1 ; accuracy or train: 71.78430939434244 accuracy or val: 75.2793959296143 accuracy and train: 48.609342175141116 accuracy and val: 51.935205559159705 
#rangeNum 2 ; accuracy or train: 87.6922787707318 accuracy or val: 90.09089145861061 accuracy and train: 74.44460632493174 accuracy and val: 77.60084002701998
#rangeNum 3 ; accuracy or train: 94.27428716449616 accuracy or val: 95.5584240130017 accuracy and train: 87.27917021106605 accuracy and val: 89.48427958988489
#rangeNum 4 ; accuracy or train: 97.00531542412949 accuracy or val: 97.67588065730777 accuracy and train: 93.21143732640276 accuracy and val: 94.51641597388961