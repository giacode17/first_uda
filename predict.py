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
pred.add_argument('--device_use', default="gpu", action="store", dest="device_use")

pred_args = pred.parse_args()

print("Image input: ", pred_args.input_img,
      "Checkpoint: ", pred_args.checkpoint,
      "TopK: ", pred_args.top_k,
      "Category names: ", pred_args.category_names,
      "Device_used: ", pred_args.device_use)
print(pred_args)



def main():
    
    load_checkpoint(pred_args.checkpoint)
    image = process_image(pred_args.input_img)
    imshow(image)
    prob_list, pred_adj = predict(image, model, pred_args.top_k, pred_args.device_use)
    
    
    with open(pred_args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    imshow(img)   
    list(cat_to_name.items())[0][1]
    names = []
    for i in pred_adj:

        j = list(cat_to_name.items())[i][1]
        names.append(j)

    print(names)
    print(prob_list)

    result = list(zip(names,prob_list))
    print(result)    
    
if __name__ == "__main__":
    main()

