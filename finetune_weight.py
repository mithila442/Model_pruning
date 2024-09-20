import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import argparse
import os
import time
from prune_weight import *
from torchvision.models import vgg16, VGG16_Weights
import io
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_size(model):
    """Calculate the size of the model in megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_in_bytes = buffer.tell()
    return size_in_bytes / (1024 ** 2)

def compact_model(model):
    """Compact model to potentially reduce memory usage."""
    model = copy.deepcopy(model)
    torch.cuda.empty_cache()
    return model

def print_model_details(model):
    for idx, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            print(f"Layer {idx}: Conv2d - in_channels: {layer.in_channels}, out_channels: {layer.out_channels}, kernel_size: {layer.kernel_size}")
        elif isinstance(layer, nn.ReLU):
            print(f"Layer {idx}: ReLU - inplace")
        elif isinstance(layer, nn.MaxPool2d):
            print(f"Layer {idx}: MaxPool2d - kernel_size: {layer.kernel_size}, stride: {layer.stride}, padding: {layer.padding}")
        elif isinstance(layer, nn.BatchNorm2d):
            print(f"Layer {idx}: BatchNorm2d - {layer.num_features}")


class ModifiedVGG16Model(nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 9)
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

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
        print(f"Model Size: {get_model_size(model):.2f} MB")

    def prune(self):
        self.model.to(device)
        print("Testing before pruning...")
        pre_pruning_accuracy = self.test()

        # Assuming calculate_pruning_ratio is a function you have defined that returns a pruning ratio
        pruning_ratio = calculate_pruning_ratio(self.model, self.test_data_loader, self.criterion, accuracy_drop_threshold=0.02)
        print(f"Calculated pruning ratio: {pruning_ratio}")

        # Assuming global_prune is a function you have defined to apply pruning to the model
        self.model = global_prune(self.model, pruning_ratio)
        self.model.to(device)  # Ensure model is on the correct device after pruning

        # Compact the model to optimize memory usage after pruning
        compact_model(model)

        print("Testing immediately after pruning...")
        immediate_post_pruning_accuracy = self.test()

        print(f"Final accepted pruning ratio: {pruning_ratio}")
        print(f"Pre-Pruning Accuracy: {pre_pruning_accuracy:.4f}")
        print(f"Immediate Post-Pruning Accuracy: {immediate_post_pruning_accuracy:.4f}")

        # Save the compacted and pruned model state
        torch.save(self.model.state_dict(), "model_pruned.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", help="Train the model")
    parser.add_argument("--prune", dest="prune", action="store_true", help="Perform global pruning and retraining")
    parser.add_argument("--train_path", type=str, default="./train", help="Path to the training dataset")
    parser.add_argument("--test_path", type=str, default="./test", help="Path to the testing dataset")
    args = parser.parse_args()

    model = ModifiedVGG16Model()
   # if torch.cuda.device_count() > 1:
       # print("Using", torch.cuda.device_count(), "GPUs!")
       # model = nn.DataParallel(model)

    model = model.to(device)
    fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        print("Starting training...")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        fine_tuner.train(optimizer=optimizer, epochs=10, phase="Pre-pruning Training")
        torch.save(model.state_dict(), "model_trained.pth")

    if args.prune:
        model.load_state_dict(torch.load("model_trained.pth", map_location=device))
        fine_tuner.prune()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        fine_tuner.train(optimizer=optimizer, epochs=10, phase="Post-pruning Training")
        torch.save(model.state_dict(), "model_pruned_and_retrained.pth")

