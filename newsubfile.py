
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse
import numpy
from train import main
from predict import predict_args

import PIL 
from PIL import Image


def load_data(data_dir):    
    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle =True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size =50)
    return trainloader, validloader

def trainer(model, save_dir, learning_rate, dropout, num_epochs, hidden_layer, device_use):    
    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("No model")
    for param in model.parameters():
        param.requires_grad = False    
  
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512*7*7, hidden_layer, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_layer, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    # change to device
    # Define loss and optimizer
    if torch.cuda.is_available() and device_use == 'gpu':
        model.cuda()
    else:
        model.cpu()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    
    epochs = int(num_epochs)
    steps = 0
    running_loss = 0
    print_every =30
    for epoch in range(epochs):
        model.train()
        for inputs, labels in iter(trainloader):
            steps += 1
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0    
                with torch.no_grad():
                    for inputs, labels in enumerate(validnloader):
                        if device_use =='gpu':
                            inputs, labels = inputs.cuda(), labels.cuda()
                        else:
                            inputs, labels = inputs.cpu(), labels.cpu()
                        
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                            f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Training setted")



    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'model': model }

    torch.save(checkpoint, save_dir)
    print("Checkpoint setted")

    return model

def load_checkpoint(filepath):
       
    model = torch.load(filepath)
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['image_datasets']
    optimizer.load_state_dict(model['optimizer'])
    return model

def process_image(image_path):
    
    for i in image_path:
        path = str(i)
    img = Image.open(i)

    adjustments = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = adjustments(img)
    return img_tensor        


def predict(image_path, model, topk=5,device_use='gpu'):

    if torch.cuda.is_available() and power=='gpu':
        model = model.cuda()

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)





