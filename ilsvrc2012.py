import scipy.io
import  shutil
import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import glob
import pandas as pd

def extract_dataset():
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.RandomCrop(180),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    dataset = ImageFolder('ILSVRC2012', transform=transform)
    print("type:",type(dataset.imgs))

    data_list = dataset.imgs
    data_list.sort(key=lambda x: x[0].split('/')[-1], reverse=False)
    #data_list = data_list[:1000]
    #dataset.imgs = data_list
    extract_num = 1000
    imgs_list = [data[0] for data in data_list[:extract_num]]
    labels_list = [data[1] for data in data_list[:extract_num]]
    print("imgs_list:{} labels_list:{}".format(imgs_list[:10],labels_list[:10]))
    #for data in data_list:
    #    print("data:",data)

    df_dict = {'image_path': imgs_list, 'label': labels_list} 
   
    df = pd.DataFrame(df_dict)
    df.to_csv('ilsvrc2012.csv',index=False)



def load_dataset():
    normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = transforms.Compose([
        transforms.RandomCrop(180),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        normalize
    ])
    dataset = ImageFolder('ILSVRC2012', transform=transform)
    #print(dataset[0])
    #print(dataset.classes[:10])  # 根据分的文件夹的名字来确定的类别
    #print(dataset.class_to_idx)  # 按顺序为这些类别定义索引为0,1...
    #print(dataset.imgs[:10])  # 返回从所有文件夹中得到的图片的路径以及其类别
    print("type:",type(dataset.imgs))

def move_valimg(val_dir='./cache/data/imagenet/val', devkit_dir='./cache/data/ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    filenames.sort()
    #filenames = filenames[:1000]
    for filename in tqdm(filenames):
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        #print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        
        #print("old_folder :{} new folder:{}".format(os.path.join(root, filename),os.path.join(output_dir, filename)))
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

if __name__ == '__main__':
    val_dir = 'ILSVRC2012'
    devkit_dir = 'ILSVRC2012_devkit_t12'
    move_valimg(val_dir=val_dir,devkit_dir=devkit_dir)
    #load_dataset()
    extract_dataset()