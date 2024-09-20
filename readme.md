# Optimizing Deep Learning Models: Weight and Filter Pruning

This repository contains the implementation of weight and filter pruning techniques applied to the VGG16 model trained on the UCI RealWaste dataset. The project aims to evaluate the effectiveness of these pruning techniques in enhancing model efficiency, particularly focusing on reducing model size, improving inference speed, and minimizing GPU memory usage without significantly compromising the accuracy.

## Project Structure

The repository is structured into two main sections:

### Filter Pruning
- `finetune_filter.py`: Contains the code for training and fine-tuning the VGG16 model using filter pruning.
- `prune_filter.py`: Implements the filter pruning logic, including the selection and removal of filters based on specified criteria.

### Weight Pruning
- `finetune_weight.py`: Manages training and fine-tuning the VGG16 model using weight pruning.
- `prune_weight.py`: Handles the application of weight pruning, including the selection and removal of weights based on their L1 norm.

## Dataset

The model is trained and evaluated using the UCI RealWaste dataset, which comprises images of waste materials categorized into nine different classes. This dataset provides a realistic scenario for testing the efficiency of neural network optimizations in image classification tasks.


## Usage

To run the training and pruning processes, use the following commands:

### For Filter Pruning:

- **Training**:
    ```bash
    python finetune_filter.py --train
    ```

- **Pruning**:
    ```bash
    python finetune_filter.py --prune
    ```

### For Weight Pruning:

- **Training**:
    ```bash
    python finetune_weight.py --train
    ```

- **Pruning**:
    ```bash
    python finetune_weight.py --prune
    ```

## Evaluation Metrics

The effectiveness of the pruning techniques is evaluated based on four key metrics:

- **Accuracy**: Measures the classification performance of the model.
- **Model Size**: Assesses the storage space required by the pruned model.
- **GPU Training Time**: Evaluates the time required to train the model on a GPU.
- **GPU Memory Usage**: Measures the amount of GPU memory utilized during training and inference.



This README provides a comprehensive guide to running and understanding the project, ensuring that users can easily replicate the results and contribute to further development.
