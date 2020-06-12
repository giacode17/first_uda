import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from newsubfile import *
import argparse
 



pred = argparse.ArgumentParser(description='Prediction')

pred.add_argument('input_img', default='./flowers/test/90/image_04431.jpg', nargs='*', action="store")
pred.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store")
pred.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
pred.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
pred.add_argument('--gpu', default="gpu", action="store", dest="device_use")

pred_args = pred.parse_args()
image_path=pred_args.input_img
topk=pred_args.top_k
device_use=pred_args.gpu
path = pred_args.checkpoint
print(pred_args)

trainloader, validloader = newsubfile.load_data()
newsubfile.load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = futils.predict(path_image, model, number_of_outputs, power)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print("Here you are")


