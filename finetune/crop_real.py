import os
import random
import numpy as np

finetune_split = 0.1

reals_path = "../real/yolo"
finetune_path = "./yolo"


real_train_x = os.listdir("../real/yolo/images/trains")
real_train_y = os.listdir("../real/yolo/labels/trains")

real_train_x.sort()
real_train_y.sort()

ids = list(range(len(real_train_x)))
random.shuffle(ids)

ids = ids[:int(len(real_train_x)*finetune_split)]


real_train_x = np.asarray(real_train_x)
real_train_y = np.asarray(real_train_y)


split_x = real_train_x[ids]
split_y = real_train_y[ids]

assert sum([x[:-4] != y[:-4] for x,y in zip(split_x, split_y)]) == 0, "Mismatched images/labels found in split datasets"


for x,y in zip (split_x, split_y):
    os.system(f'cp {reals_path}/images/trains/{x} {finetune_path}/images/trains/{x}')
    os.system(f'cp {reals_path}/labels/trains/{y} {finetune_path}/labels/trains/{y}')