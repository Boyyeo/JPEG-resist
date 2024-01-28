import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from absl import flags, app
import random
from torchvision import models
import os
from attacks import  fgsm, fgsm_jpeg
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

    train_dataset = CustomImageDataset(csv_file='ilsvrc2012.csv',transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.bs,shuffle=True, num_workers=8)
    model_name = 'resnet50'
    model = getattr(models, model_name)(pretrained=True).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    jpeg_fn_25 = DiffJPEG(height=224, width=224, differentiable=False, quality=25)
    jpeg_fn_50 = DiffJPEG(height=224, width=224, differentiable=False, quality=50)
    jpeg_fn_75 = DiffJPEG(height=224, width=224, differentiable=False, quality=75)
    jpeg_fn_list = [jpeg_fn_25,jpeg_fn_50,jpeg_fn_75]
    count = 0 

    result = {'ori':0,'adv':0,'resist-adv':0,'adv-jpeg-25':0,'adv-jpeg-50':0,'adv-jpeg-75':0,'resist-adv-jpeg-25':0,'resist-adv-jpeg-50':0,'resist-adv-jpeg-75':0}
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        #labels = torch.argmax(model(Normalize(images))).reshape(1)

        if FLAGS.target:
            target_labels = (labels + 500) % 1000

        attacked_images = fgsm(model,criterion,images,labels if not FLAGS.target else target_labels,eps=FLAGS.eps,alpha=FLAGS.alpha,num_iter=FLAGS.num_iter,target=FLAGS.target)
        resisted_attacked_images = fgsm_jpeg(model,criterion,images,labels if not FLAGS.target else target_labels,eps=FLAGS.eps,alpha=FLAGS.alpha,num_iter=FLAGS.num_iter,target=FLAGS.target)
        
        attacked_images_jpeg_list, resisted_attacked_images_jpeg_list = [], []
        for jpeg_fn in jpeg_fn_list:
            jpeg_img = jpeg_fn(attacked_images)
            resisted_jpeg_img = jpeg_fn(resisted_attacked_images)
            attacked_images_jpeg_list.append(jpeg_img)
            resisted_attacked_images_jpeg_list.append(resisted_jpeg_img)

        attacked_images_jpeg_list = torch.cat(attacked_images_jpeg_list,dim=0)
        resisted_attacked_images_jpeg_list = torch.cat(resisted_attacked_images_jpeg_list,dim=0)
        # Total 9 images concat in order
        image_array = torch.cat([images,attacked_images,resisted_attacked_images,attacked_images_jpeg_list,resisted_attacked_images_jpeg_list],dim=0)


        with torch.no_grad():
            image_array = Normalize(image_array)
            output = model(image_array)
            preds = torch.argmax(output,dim=1)
        
        i = 0
        for k, v in result.items():
            result[k] += int((preds[i]==labels if not FLAGS.target else preds[i]==target_labels).sum())
            i += 1

        if FLAGS.save_attacked_image:
            os.makedirs('tmp/',exist_ok=True)
            save_image(image_array,fp='tmp/adv_images_{}.png'.format(str(count).zfill(4)))
       
        count += 1
       
       
    for k, v in result.items():
            result[k] = round(result[k]/count,6)
            print("Accuracy [{}]: {}".format(k,result[k]))
  
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