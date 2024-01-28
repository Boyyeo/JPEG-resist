## Unofficial Implementation of JPEG-resist Adversarial Images

### Dataset 
```
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
mkdir ILSVRC2012
unzip ILSVRC2012_img_val.tar -d ILSVRC2012
unzip /ILSVRC2012_devkit_t12.tar.gz
python ilsvrc2012.py (ilsvrc2012.csv (with the first 1000 images annotations) will be generated)
```

### Evaluation
#### Target I-FGSM (eps:9/255)
```
CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 9 --num_iter 10 --alpha 9 --target 

```

#### UnTarget I-FGSM (eps:3/255)
```
CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 3 --num_iter 10 --alpha 3
```

#### Untarget FGSM (eps:9/255)
```
CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 9 --num_iter 1 --alpha 9

```

