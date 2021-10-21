import os

# unet 0429 subsequence analysis different dataset
# os.system("python main.py --dataset Sanitation --subseq 96")
# os.system("python main.py --dataset Sanitation --subseq 96")
# os.system("python main.py --dataset Smartforceps --net test_model_2D")
# os.system("python main.py --dataset Smartforceps --subseq 200 --net lstm_model")
os.system("python main.py --dataset Smartforceps --subseq 200 --net inception_time")
# os.system("python main.py --dataset Sanitation --subseq 160")
# os.system("python main.py --dataset Sanitation --subseq 288")


# UNET different block on different datasets

# os.system("python main.py --dataset Sanitation --subseq 224 --block 4")
# os.system("python main.py --dataset Sanitation --subseq 224 --block 3")
# os.system("python main.py --dataset Sanitation --subseq 224 --block 2")

