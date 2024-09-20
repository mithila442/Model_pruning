import torch
import torch.nn as nn
from torch.nn.utils import prune
import copy
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(model, test_data_loader, criterion):
    """Evaluate the model's performance on a test dataset."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (batch, label) in enumerate(test_data_loader):
            batch, label = batch.to(device), label.to(device)
            output = model(batch)
            loss += criterion(output, label).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(label).sum().item()
            total += label.size(0)
    accuracy = float(correct) / total
    return accuracy, loss / total

def global_prune(model, pruning_ratio):
    """Apply global unstructured L1 pruning to all convolutional layers."""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            if module.bias is not None:
                parameters_to_prune.append((module, 'bias'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    # Make pruning permanent
    for module, name in parameters_to_prune:
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)
            if module.bias is not None and hasattr(module.bias, "bias_orig"):
                prune.remove(module, 'bias')
    return model

def calculate_pruning_ratio(model, test_loader, criterion, initial_pruning_ratio=0.01, accuracy_drop_threshold=0.02, max_pruning_ratio=0.7, min_increment=0.01):
    """Determine the optimal pruning ratio that does not significantly degrade model accuracy."""
    original_accuracy, _ = test_model(model, test_loader, criterion)
    print(f"Original model accuracy: {original_accuracy:.4f}")
    pruning_ratio = initial_pruning_ratio
    while pruning_ratio <= max_pruning_ratio:
        pruned_model = copy.deepcopy(model)
        pruned_model = global_prune(pruned_model, pruning_ratio)
        pruned_accuracy, _ = test_model(pruned_model, test_loader, criterion)
        print(f"Testing pruning at ratio {pruning_ratio:.4f}: Pruned model accuracy {pruned_accuracy:.4f}")
        if original_accuracy - pruned_accuracy > accuracy_drop_threshold:
            print(f"Stopping: Accuracy drop greater than threshold after pruning at ratio {pruning_ratio:.4f}")
            break
        pruning_ratio += min_increment
    final_pruning_ratio = pruning_ratio - min_increment
    print(f"Final accepted pruning ratio: {final_pruning_ratio:.4f} (Accuracy drop within threshold)")
    return final_pruning_ratio
