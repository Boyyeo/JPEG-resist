CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 9 --num_iter 10 --alpha 9 --target #Target I-FGSM
CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 9 --num_iter 10 --alpha 9 #UnTarget I-FGSM
CUDA_VISIBLE_DEVICES=3 python run.py --bs 1 --eps 9 --num_iter 1 --alpha 9  #Untarget FGSM