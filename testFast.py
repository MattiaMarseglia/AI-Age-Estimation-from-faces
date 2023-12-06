import os, sys, cv2, csv, argparse, random, torch
import numpy as np
import torch
#This code is to try a single regressor with our backbone
import os
import cv2
import dlib
import copy
import torch
import random
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
# from torchsummary import summary #network summary
from torchvision.models import Inception_V3_Weights, Inception3

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

        sample = {'image': image, 'age': age, 'name':self.df.iloc[idx,0]}
        return sample

class SimpleRegressorModel(nn.Module):
    def __init__(self, Backbone, LastRegressor):
        super(SimpleRegressorModel, self).__init__()
        self.Backbone = Backbone
        self.LastRegressor = LastRegressor

        
    def forward(self, image):
        featureVect1 = self.Backbone(image)
        output = self.LastRegressor(featureVect1)
        return output

class eoClassifierModel(nn.Module):
    def __init__(self, Backbone, LasteoClassifierModel):
        super(eoClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.LasteoClassifierModel = LasteoClassifierModel
        
    def forward(self, image):
        featureVect = self.Backbone(image)
        output = self.LasteoClassifierModel(featureVect)
        return output

class RegressorModel(nn.Module):
    def __init__(self, Classifier1, Classifier2, RegressorHead, LastRegressor):
        super(RegressorModel, self).__init__()
        self.Classifier1 = Classifier1
        self.Classifier2 = Classifier2
        self.RegressorHead = RegressorHead
        self.LastRegressor = LastRegressor

        
    def forward(self, image):

        C1 = self.Classifier1.Backbone
        C2 = self.Classifier2.Backbone
        x = C1.conv2d_1a(image)
        x = C1.conv2d_2a(x)
        x = C1.conv2d_2b(x)
        x = C1.maxpool_3a(x)
        x = C1.conv2d_3b(x)
        x = C1.conv2d_4a(x)
        x = C1.conv2d_4b(x)

        fetaureSplitClassandReg = C1.repeat_1(x)
        xToC = C1.mixed_6a(fetaureSplitClassandReg)
        featurePreSpilt = C1.repeat_2(xToC)

        x1 = C1.mixed_7a(featurePreSpilt)
        x1 = C1.repeat_3(x1)
        x1 = C1.block8(x1)
        x1 = C1.avgpool_1a(x1)
        x1 = C1.dropout(x1)
        x1 = C1.last_linear(x1.view(x1.shape[0], -1))
        featureVect1 = C1.last_bn(x1)
        classPredict1 = self.Classifier1.LasteoClassifierModel(featureVect1)

        x2 = C2.mixed_7a(featurePreSpilt)
        x2 = C2.repeat_3(x2)
        x2 = C2.block8(x2)
        x2 = C2.avgpool_1a(x2)
        x2 = C2.dropout(x2)
        x2 = C2.last_linear(x2.view(x2.shape[0], -1))
        featureVect2 = C2.last_bn(x2)
        classPredict2 = self.Classifier2.LasteoClassifierModel(featureVect2)

        prob1 = nn.Softmax(dim=1)(classPredict1)
        prob2 = nn.Softmax(dim=1)(classPredict2)
        
        preds1 = torch.argmax(prob1,dim=1).unsqueeze(1)
        preds2 = torch.argmax(prob2,dim=1).unsqueeze(1)
        
        
        tensor1 = torch.zeros((prob1.size()[0], 81), dtype=torch.float32).to(device)

        # Create a mask for indices within +-4 of the second vector's value
        mask = torch.arange(81).repeat(prob1.size()[0], 1).to(device)
        mask1 = (mask >= preds1 - 4) & (mask <= preds1 + 4)
        # Set the mask to 1 in the first 
        tensor1[mask1] = 1

        tensor2 = torch.zeros((prob1.size()[0], 81), dtype=torch.float32).to(device)

        # Create a mask for indices within +-4 of the second vector's value

        mask2 = (mask >= preds2 - 4) & (mask <= preds2 + 4)
        # Set the mask to 1 in the first     
        tensor2[mask2] = 1

        allProb = torch.cat((tensor1, tensor2), dim=1)
        x3 = self.RegressorHead[0](fetaureSplitClassandReg)
        x3 = self.RegressorHead[1](x3)
        x3 = self.RegressorHead[2](x3)
        x3 = self.RegressorHead[3](x3)
        x3 = self.RegressorHead[4](x3)
        x3 = self.RegressorHead[5](x3)
        x3 = self.RegressorHead[6](x3)
        x3 = self.RegressorHead[7](x3.view(x3.shape[0], -1))
        x3 = self.RegressorHead[8](x3)
        featureReg = F.normalize(x3, p=2, dim=1)
        
        output = self.LastRegressor(torch.cat((featureReg, allProb), dim=1))
        return output
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

# modelPath = ('/user/2022_va_gr12/landmarks5.dat')
modelPath = ('./utils/landmarks5.dat')

allignment = allignment(modelPath)

class face_alignment_val(object):

    def __call__(self,img): 
        
        img = np.array(img) 

        img = allignment.allignImage(img).astype(np.uint8)
        
        return preprocess_input(img)

def createModel():
        #BACKBONE:
    from facenet_pytorch import InceptionResnetV1

    # For a model pretrained on VGGFace2
    backbone1 = InceptionResnetV1(pretrained='vggface2')
    backbone2 = InceptionResnetV1(pretrained='vggface2')
    backboneReg = InceptionResnetV1(pretrained='vggface2')


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

    ##REGRESSOR

    #REGRESSOR TO LOAD 
    num_ftrsReg=num_ftrs
    LastRegressor = nn.Sequential(
            nn.Linear(num_ftrsReg, num_ftrsReg),
            nn.AvgPool1d(2),
            nn.Dropout(0.2),
            nn.Linear(int(num_ftrsReg/2), 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.AvgPool1d(2),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Linear(4, 1), #regression
    )

    regModel = SimpleRegressorModel(backboneReg, LastRegressor)


    regModel.LastRegressor[0] = nn.Linear(num_ftrsReg + 162, num_ftrsReg)

    RegressorHead=copy.deepcopy(nn.Sequential(
            regModel.Backbone.mixed_6a,
            regModel.Backbone.repeat_2,
            regModel.Backbone.mixed_7a,
            regModel.Backbone.repeat_3,
            regModel.Backbone.block8,
            regModel.Backbone.avgpool_1a,
            regModel.Backbone.dropout,
            regModel.Backbone.last_linear,
            regModel.Backbone.last_bn
        ))

    #CLASSIFIER MODEL
    classifier1 = eoClassifierModel(backbone1, ClassifierModel1)
    classifier2 = eoClassifierModel(backbone2, ClassifierModel2)

    #FINAL MODEL
    model = RegressorModel(classifier1, classifier2, RegressorHead, regModel.LastRegressor)


    checkpoint1 = torch.load('./utils/TotalModelPrimoTrainingFinalOnlyLastRegSThirdWayArgMax.pt')
    checkpoint1.keys()
    model.load_state_dict(checkpoint1)

    return model

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--data", type=str, default='foo_test.csv', help="Dataset labels")
    parser.add_argument("--images", type=str, default='foo_test/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results.csv', help="CSV file of the results")
    args = parser.parse_args()
    return args

args = init_parameter()

print(torch.cuda.device_count())

CUDA_VISIBLE_DEVICES=0
torch.cuda.set_per_process_memory_fraction(0.16, CUDA_VISIBLE_DEVICES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = createModel()    
model.to(device)
model.eval()

data_transforms_val = transforms.Compose([
transforms.Resize((160,160)), #299 299
face_alignment_val(),
transforms.ToTensor(),
transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])


df = pd.read_csv(
    args.data,
    names=["image", "age"],dtype={'image':'str','age':'float'})


TesteDataSet = dataFrameDataset(df,args.images,data_transforms_val)
batch_size = 512
# create batches
testSet = DataLoader(TesteDataSet, shuffle=False,batch_size=batch_size,num_workers=15)


def evaluate_aar(loader, model):
    # initialize metric
    model.eval()

    my_dict = pd.DataFrame()
    with torch.no_grad():
        
        for sample_batched in tqdm(loader):
            inputs = sample_batched['image'].float().to(device)
            name = sample_batched['name']


            outputs = model(inputs)

            for idx,out in enumerate(outputs):
                tmp_dict = pd.DataFrame({'name':name[idx],'value':out.item()},index=[1])
                my_dict = pd.concat([my_dict, tmp_dict])

    my_dict.to_csv(args.results, index=False, header=False)

            

evaluate_aar(testSet,model)