import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from absl import flags, app
import random
from torchvision import models
import os
from torchvision.datasets import ImageFolder
from attacks import fgm, fgm_jpeg_resist, fgsm, fgsm_jpeg
from DiffJPEG.DiffJPEG import DiffJPEG
from tqdm import tqdm
from torchvision.utils import save_image
from dataset import CustomImageDataset
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_integer('bs', 128, "batch size")
flags.DEFINE_integer('epoch', 500, "number of epochs")
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_string('name', "experiment", "Folder output sample_image")
flags.DEFINE_float('eps', 9, "epsilon hyperparameter for FGSM")
flags.DEFINE_integer('num_iter', 1, "iter hyperparameter for FGSM")
flags.DEFINE_float('alpha', 9, "alpha hyperparameter for FGSM")
flags.DEFINE_boolean('save_attacked_image', False, "whether to save the adversarial images")
flags.DEFINE_boolean('target', False, "whether targeted FGSM")

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    Normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = T.Compose(
        [
            T.Resize(size=(224,224)),
            T.ToTensor(),
       ])

    #train_dataset = ImageFolder('ILSVRC2012', transform=train_transform)
    train_dataset = CustomImageDataset(csv_file='ilsvrc2012.csv',transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.bs,shuffle=True, num_workers=8)
    model_name = 'resnet50'
    model = getattr(models, model_name)(pretrained=True).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    jpeg_fn_25 = DiffJPEG(height=224, width=224, differentiable=False, quality=25)
    jpeg_fn_50 = DiffJPEG(height=224, width=224, differentiable=False, quality=50)
    jpeg_fn_75 = DiffJPEG(height=224, width=224, differentiable=False, quality=75)

    count = 0 
    original_correct = 0 
    adv_correct = 0
    jpeg_adv_correct = 0
    jpeg_resist_adv_correct = 0   
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        ########### TODO: #######################
        attacked_images = fgm(model,criterion,images,labels,ord,eps=FLAGS.eps,eps_iter=FLAGS.eps/FLAGS.num_iter,nb_iter=FLAGS.num_iter,clip_min=-1,clip_max=1,targeted=None)
        attacked_images_jpeg_resist = fgm_jpeg_resist(model,criterion,images,labels,ord,eps=FLAGS.eps,eps_iter=FLAGS.eps/FLAGS.num_iter,nb_iter=FLAGS.num_iter,clip_min=-1,clip_max=1,targeted=None)
        if FLAGS.target:
            target_labels = (labels + 500) % 1000

        #attacked_images, _, _  = fgsm(model,criterion,images,labels if not FLAGS.target else target_labels,eps=FLAGS.eps,alpha=FLAGS.alpha,num_iter=FLAGS.num_iter,target=FLAGS.target)
        #attacked_images_jpeg_resist, _, _  = fgsm_jpeg(model,criterion,images,labels if not FLAGS.target else target_labels,eps=FLAGS.eps,alpha=FLAGS.alpha,num_iter=FLAGS.num_iter,target=FLAGS.target)
        jpeg_attacked_images = jpeg_fn_25(attacked_images).cuda()
        jpeg_resist_attacked_images = jpeg_fn_25(attacked_images_jpeg_resist).cuda()

        with torch.no_grad():
            images = Normalize(images)
            attacked_images = Normalize(attacked_images)
            jpeg_attacked_images = Normalize(jpeg_attacked_images)
            jpeg_resist_attacked_images = Normalize(jpeg_resist_attacked_images)

            pred = np.argmax(model(images).data.cpu().numpy())
            pred_adv = np.argmax(model(attacked_images).data.cpu().numpy())
            pred_jpeg_adv = np.argmax(model(jpeg_attacked_images).data.cpu().numpy())
            pred_jpeg_resist_adv = np.argmax(model(jpeg_resist_attacked_images).data.cpu().numpy())

        if FLAGS.save_attacked_image:
            os.makedirs('tmp/',exist_ok=True)
            save_image(jpeg_attacked_images,fp='tmp/jpeg-attack_{}.png'.format(str(count).zfill(4)))
            save_image(jpeg_resist_attacked_images,fp='tmp/jpeg-resist_{}.png'.format(str(count).zfill(4)))
            save_image(images,fp='tmp/ori_{}.png'.format(str(count).zfill(4)))
            save_image(attacked_images,fp='tmp/attack_{}.png'.format(str(count).zfill(4)))

        count += 1
        original_correct += int((pred==labels if not FLAGS.target else pred==target_labels).sum())
        adv_correct += int((pred_adv==labels if not FLAGS.target else pred_adv==target_labels).sum())
        jpeg_adv_correct += int((pred_jpeg_adv==labels if not FLAGS.target else pred_jpeg_adv==target_labels).sum())
        jpeg_resist_adv_correct += int((pred_jpeg_resist_adv==labels if not FLAGS.target else pred_jpeg_resist_adv==target_labels).sum())
        if count >= 100:
            break
       
    
    original_correct = round(original_correct/count,6)
    adv_correct = round(adv_correct/count,6)
    jpeg_adv_correct = round(jpeg_adv_correct/count,6)
    jpeg_resist_adv_correct = round(jpeg_resist_adv_correct/count,6)
    print("Accuracy: Ori-{} Adv-{} JPEG-Adv:{} JPEG-Resist_Adv:{}".format(original_correct,adv_correct,jpeg_adv_correct,jpeg_resist_adv_correct))
    #Accuracy: Ori-0.79 Adv-0.23 JPEG-Adv:0.3 JPEG-Resist_Adv:0.04

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(argv):
    set_seed(FLAGS.seed)
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    run = 0
    while os.path.exists(FLAGS.out_dir + FLAGS.name + str(run) + "/"):
        run += 1
    FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + str(run) + "/"
    os.mkdir(FLAGS.out_dir)
    os.mkdir(FLAGS.out_dir + "checkpoint")
    train()


if __name__ == '__main__':
    app.run(main)