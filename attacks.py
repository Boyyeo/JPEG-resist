import torch
from tqdm import tqdm 
from torch.autograd import Variable
import numpy as np
from DiffJPEG.DiffJPEG import DiffJPEG

def batchwise_norm(x, ord):
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

def fgm(model, criterion, images, labels, ord, eps, eps_iter, nb_iter, clip_min, clip_max, targeted=None):
    orig_images = images.clone().detach()
    images = Variable(images.clone(), requires_grad=True)

    for i in range(nb_iter):
        out = model(images)
        loss = criterion(out, labels).mean()
        loss.backward()
        
        grad_images = images.grad.data
        if ord == float('inf'):
            grad_images = torch.sign(grad_images)
        elif ord in (1, 2):
            grad_images /= batchwise_norm(grad_images, ord)

        # Take eps_iter step
        grad_images *= eps_iter
        grad_images = torch.clamp((images.data + grad_images) - orig_images, min=-eps/255.0, max=eps/255.0)

        if targeted:
            images.data = images.data - grad_images 
        else:
            images.data = images.data + grad_images 

        # Ensure |images - orig_images| < eps
        #factor = torch.maximum(batchwise_norm(images - orig_images, ord) / eps, torch.tensor(1))
        #images.data = images.data/factor
        #images.data = images.data + orig_images * (1 - 1./factor)

        #images.data = torch.clamp(images.data,min=clip_min,max=clip_max)
        #images.grad.data.zero_()

    return images


def fgm_jpeg_resist(model, criterion, images, labels, ord, eps, eps_iter, nb_iter, clip_min, clip_max, targeted=None):
    orig_images = images.clone().detach()
    images = Variable(images.clone(), requires_grad=True)

    for i in range(nb_iter):
        jpeg_inp = make_diff_jpeg(images)
        out = model(jpeg_inp)
        loss = criterion(out, labels.repeat(out.shape[0])).mean()
        loss.backward()
        
        grad_images = images.grad.data
        if ord == float('inf'):
            grad_images = torch.sign(grad_images)
        elif ord in (1, 2):
            grad_images /= batchwise_norm(grad_images, ord)

        # Take eps_iter step
        grad_images *= eps_iter
        grad_images = torch.clamp((images.data + grad_images) - orig_images, min=-eps/255.0, max=eps/255.0)

        if targeted:
            images.data = images.data - grad_images 
        else:
            images.data = images.data + grad_images 

        # Ensure |images - orig_images| < eps
        #factor = torch.maximum(batchwise_norm(images - orig_images, ord) / eps, torch.tensor(1))
        #images.data = images.data/factor
        #images.data = images.data + orig_images * (1 - 1./factor)

        #images.data = torch.clamp(images.data,min=clip_min,max=clip_max)
        images.grad.data.zero_()

    return images


def fgsm_jpeg(model, criterion, images, labels, eps, alpha, num_iter, target=None):
    inp = Variable(images, requires_grad=True)
    orig = images.float()
    out = model(inp)
    pred = np.argmax(out.data.cpu().numpy())

    for i in range(num_iter):
    
        ##############################################################
        jpeg_inp = make_diff_jpeg(inp)
        out = model(jpeg_inp)
        #target_labels = Variable(torch.Tensor(jpeg_inp.shape[0] * [float(pred)]).to(images.device).long())
        target_labels = labels.repeat(out.shape[0])

        loss = criterion(out, target_labels)
        if target:
            loss = -1 * loss
        loss.backward()
        # this is the method
        perturbation = (alpha/num_iter) * torch.sign(inp.grad.data)
        perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
        inp.data = orig + perturbation

        inp.grad.data.zero_()
        ################################################################

    pred_adv = np.argmax(model(inp).data.cpu().numpy())
    #print("Iter [%3d/%3d]:  Prediction: %s" %(i, num_iter, classes[pred_adv].split(',')[0]))
    return inp, pred, pred_adv

def fgsm(model, criterion, images, labels, eps, alpha, num_iter, target=None):
    inp = Variable(images, requires_grad=True)
    orig = images.float()
    out = model(inp)
    pred = np.argmax(out.data.cpu().numpy())

    for i in range(num_iter):
    
        ##############################################################
        out = model(inp)
        #loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(images.device).long()))
        loss = criterion(out, labels)
        if target:
            loss = -1 * loss
        loss.backward()
   
        # this is the method
        perturbation = (alpha/num_iter) * torch.sign(inp.grad.data)
        perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
        inp.data = orig + perturbation

        inp.grad.data.zero_()
        ################################################################

    pred_adv = np.argmax(model(inp).data.cpu().numpy())
    #print("Iter [%3d/%3d]:  Prediction: %s" %(i, num_iter, classes[pred_adv].split(',')[0]))
    return inp, pred, pred_adv