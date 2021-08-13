import os

# unet subsequence analysis different dataset
os.system("python main.py --dataset Smartforceps --subseq 96")

# UNET different block on different datasets

# os.system("python main.py --dataset Smartforceps --subseq 224 --block 4")
# os.system("python main.py --dataset Smartforceps --subseq 224 --block 3")
# os.system("python main.py --dataset Smartforceps --subseq 224 --block 2")

# FCN

# os.system("python main.py --dataset Smartforceps --net fcn")




