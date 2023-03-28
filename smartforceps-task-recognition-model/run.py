import os

# unet 0429 subsequence analysis different dataset
# os.system("python main.py --dataset Smartforceps --subseq 200 --net lstm_model")
# os.system("python main.py --dataset Smartforceps --subseq 200 --net inception_time")
# os.system("python main.py --dataset Smartforceps --subseq 96 --net inception_time")

os.system("python main_with_crossval.py --dataset Smartforceps --subseq 200 --net inception_time")

