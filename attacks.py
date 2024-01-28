import torch
from tqdm import tqdm 
from torch.autograd import Variable
import numpy as np
from DiffJPEG.DiffJPEG import DiffJPEG
import torch.nn.functional as F
from torchvision import transforms as T

Normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def batchwise_norm(x, ord=float('inf')):
    dims = tuple(range(1, len(x.shape)))
    if ord == float('inf'):
        return torch.amax(torch.abs(x), dim=dims, keepdim=True)
    elif ord == 1:
        return torch.sum(torch.abs(x), dim=dims, keepdim=True)
    elif ord == 2:
        return torch.sqrt(torch.sum(torch.square(x), dim=dims, keepdim=True))
    else:
        raise ValueError(ord)

def make_diff_jpeg(images):
    jpeg_diff_25_fn = DiffJPEG(height=224, width=224, differentiable=True, quality=25)
    jpeg_diff_50_fn = DiffJPEG(height=224, width=224, differentiable=True, quality=50)
    jpeg_diff_75_fn = DiffJPEG(height=224, width=224, differentiable=True, quality=75)
    images_jpeg_25 = jpeg_diff_25_fn(images).cuda()
    images_jpeg_50 = jpeg_diff_50_fn(images).cuda()
    images_jpeg_75 = jpeg_diff_75_fn(images).cuda()
    jpeg_concat_images = torch.cat([images,images_jpeg_25,images_jpeg_50,images_jpeg_75],dim=0)
    return jpeg_concat_images

def fgsm_jpeg(model, criterion, images, labels, eps, alpha, num_iter, target=None):
    inp = Variable(images, requires_grad=True)
    orig = images.float()

    for i in range(num_iter):
        jpeg_inp = make_diff_jpeg(inp)
        jpeg_inp = Variable(jpeg_inp, requires_grad=True)

        out = model(Normalize(jpeg_inp))
        target_labels = labels.repeat(out.shape[0])
        
        loss = criterion(out, target_labels)
        loss_weight = F.softmax(loss)
    
        if target:
            loss = -1 * loss
        loss.mean().backward()

        jpeg_inp_grad = jpeg_inp.grad.data
        grad = loss_weight[0] * jpeg_inp_grad[0] + loss_weight[1] * jpeg_inp_grad[1] + loss_weight[2] * jpeg_inp_grad[2] + loss_weight[3] * jpeg_inp_grad[3]
        
        perturbation = ((alpha/255.)/num_iter) * torch.sign(grad)
        # Ensure |images - orig_images| < eps
        inp.data = inp.data + perturbation 
        factor = torch.maximum(batchwise_norm(inp.data - orig) / (eps/255.), torch.tensor(1))
        inp.data = inp.data/factor
        inp.data = inp.data + orig * (1 - 1./factor)
        inp.data = torch.clamp(inp.data , min=0, max=1)

        jpeg_inp.grad.data.zero_()
        

    return inp

def fgsm(model, criterion, images, labels, eps, alpha, num_iter, target=None):
    inp = Variable(images, requires_grad=True)
    orig = images.float()

    for i in range(num_iter):
        out = model(Normalize(inp))
        loss = criterion(out, labels).mean()
        if target:
            loss = -1 * loss
        loss.mean().backward()
   
        perturbation = ((alpha/255.)/num_iter) * torch.sign(inp.grad.data)
    
        # Ensure |images - orig_images| < eps
        inp.data = inp.data + perturbation 
        factor = torch.maximum(batchwise_norm(inp.data - orig) / (eps/255.), torch.tensor(1))
        inp.data = inp.data/factor
        inp.data = inp.data + orig * (1 - 1./factor)
        inp.data = torch.clamp(inp.data , min=0, max=1)

        inp.grad.data.zero_()

    return inp