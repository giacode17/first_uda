import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from newsubfile import load_data, trainer




tr = argparse.ArgumentParser(description='Training')

tr.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
tr.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
tr.add_argument('--arch', action="store", dest="model", default="vgg16")
tr.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
tr.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
tr.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
tr.add_argument('--hidden_layer', dest="hidden_layer", action="store", type=int, default=4096)
tr.add_argument('--device_use', dest="device_use", action="store", default="gpu")

#tr.add_argument('--fixed_model', dest="fixed_model", action="store", default="vgg16", type = str)



tr_args = tr.parse_args()

print("Img_Directory: ", tr_args.data_dir, 
      "Save_Directory: ", tr_args.save_dir, 
      "Model: ", tr_args.model, 
      "Learning Rate: ", tr_args.learning_rate,
      "Drop_out: ", tr_args.dropout,
      "Epochs: ", tr_args.epochs, 
      "Hidden_layer: ", tr_args.hidden_layer, 
      "Device_used :", tr_args.device_use )

    


def main():
    print(tr_args.model)
    trainer(tr_args.model, tr_args.data_dir, tr_args.save_dir, tr_args.learning_rate, tr_args.dropout, tr_args.epochs, tr_args.hidden_layer, tr_args.device_use)
    
# Call to get_input_args function to run the program
if __name__ == "__main__":
    main()


 


