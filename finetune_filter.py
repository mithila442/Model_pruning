import copy
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune_filter import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
from torchvision.models import vgg16, VGG16_Weights
import io
import torchprofile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 9))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank(activation_index))
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, activation_index):
        # Returns a partial function
        # as the callback function
        def hook(grad):
            activation = self.activations[activation_index]
            # print((activation * grad).shape)
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True).\
                 sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
            #    sum(dim=2).sum(dim=3)[0, :, 0, 0].data

            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(2)
                          * activation.size(3))

            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = \
                    torch.FloatTensor(activation.size(1)).zero_().cuda()

            self.filter_ranks[activation_index] += values
            self.grad_index += 1
        return hook

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append(
                    (self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            # Move tensor to CPU before using numpy functions
            v = v.cpu()
            # Perform the numpy operation on a tensor that is now on the CPU
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(
                filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, label in self.test_data_loader:
                batch, label = batch.to(device), label.to(device)
                output = self.model(batch)
                pred = output.max(1)[1]
                correct += pred.eq(label).sum().item()
                total += label.size(0)
        accuracy = float(correct) / total
        return accuracy

    def train(self, optimizer=None, epochs=10, phase="Training"):
        if optimizer is None:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        start_time = time.time()
        total_mem_allocated = 0
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in self.train_data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # GPU memory in MB
            total_mem_allocated += mem_allocated
            print(f"{phase} - Epoch {epoch + 1} Accuracy: {self.test():.4f}")
        end_time = time.time()
        print(f"{phase} - Training Time: {end_time - start_time:.2f} seconds")
        print(f"Total GPU Memory Usage: {total_mem_allocated:.2f} MB")
        print(f"Model Size: {self.get_model_size(model):.2f} MB")

    def get_model_size(self, model):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 ** 2)  # Convert bytes to MB

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(),
                             label.cuda(), rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(rank_filters=True)

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        self.test()

        self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) /
                         num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print("Number of prunning iterations to reduce 67% filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(
                num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(
                    model, layer_index, filter_index)

            self.model = model.cuda()

            message = str(100 * float(self.total_num_filters()) /
                          number_of_filters) + "%"
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
            self.train(optimizer, epochs=10)

        # print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epochs=15)
        torch.save(model, "model_prunned")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    
    model = ModifiedVGG16Model()
   # if torch.cuda.device_count() > 1:
       # print("Using", torch.cuda.device_count(), "GPUs!")
       # model = nn.DataParallel(model)

    model = model.to(device)
    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)
    if args.train:
        model = ModifiedVGG16Model().cuda()
    elif args.prune:
        model = torch.load("model").cuda()

    fine_tuner = PrunningFineTuner_VGG16(
        args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epochs=10)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()

