#https://github.com/sarathknv/adversarial-examples-pytorch/tree/master/iterative
""" Basic Iterative Method (Targeted and Non-targeted)
    Paper link: https://arxiv.org/abs/1607.02533

    Controls:
        'esc' - exit
         's'  - save adversarial image
      'space' - pause
"""
import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import argparse
from imagenet_labels import classes
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/goldfish.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--target', type=int, required=False, default=None, help='target label')

args = parser.parse_args()
image_path = args.img
model_name = args.model
y_true = args.y
target = args.target
gpu = args.gpu

IMG_SIZE = 224

print('Iterative Method')
print('Model: %s' %(model_name))
print()


# break loop when parameters are changed
break_loop = False


# load image and reshape to (3, 224, 224) and RGB (not BGR)
# preprocess as described here: http://pytorch.org/docs/master/torchvision/models.html
orig = cv2.imread(image_path)[..., ::-1]
orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
img = orig.copy().astype(np.float32)
#perturbation = np.empty_like(orig)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean)/std
img = img.transpose(2, 0, 1)


# load model
model = getattr(models, model_name)(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()

device = 'cuda' if gpu else 'cpu'


# prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
orig = torch.from_numpy(img).float().to(device).unsqueeze(0)

out = model(inp)
pred = np.argmax(out.data.cpu().numpy())

print('Prediction before attack: %s' %(classes[pred].split(',')[0]))
if target is not None:
    pred = target
    print('Prediction target attack: %s' %(classes[target].split(',')[0]))

eps = 3
alpha = 10
num_iter = 50

print('eps [%d]' %(eps))
print('Iter [%d]' %(num_iter))
print('alpha [1]')
print('-'*20)
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)

for i in range(num_iter):

    ##############################################################
    out = model(inp)
    loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
    if target is not None:
        loss = -1 * loss
    loss.backward()

    # this is the method
    perturbation = (alpha/255.0) * torch.sign(inp.grad.data)
    perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
    inp.data = orig + perturbation

    inp.grad.data.zero_()
    ################################################################

    pred_adv = np.argmax(model(inp).data.cpu().numpy())

    print("Iter [%3d/%3d]:  Prediction: %s" %(i, num_iter, classes[pred_adv].split(',')[0]))


    # deprocess image
    adv = inp.data.cpu().numpy()[0]
    pert = (adv-img).transpose(1,2,0)
    adv = adv.transpose(1, 2, 0)
    adv = (adv * std) + mean
    adv = adv * 255.0
    adv = adv[..., ::-1] # RGB to BGR
    adv = np.clip(adv, 0, 255).astype(np.uint8)
    cv2.imwrite('adv_img.png',adv)
    pert = pert * 255
    pert = np.clip(pert, 0, 255).astype(np.uint8)

           

            

