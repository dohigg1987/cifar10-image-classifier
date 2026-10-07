# CIFAR-10 Image Classifier (PyTorch)

**Author:** Dennis O'Higgins

A convolutional neural network built in PyTorch to classify the ten CIFAR-10 object classes, written for a build-or-buy decision at a fictional self-driving car start-up that was considering buying Detectocorp's algorithm (70% accuracy).

## Result

| Model | Test accuracy |
|---|---|
| Detectocorp (claimed) | 70.0% |
| **This network** | **91.82%** (9,182 of 10,000 test images) |

The model has about 2.2 million parameters and trained in about 10 minutes on a single NVIDIA T4 GPU (Google Colab). The recommendation in the notebook is to **build** in-house.

## Approach

- **Data:** CIFAR-10 loaded with `torchvision.datasets.CIFAR10`; 45,000 training images, 5,000 held out for validation, 10,000 test images.
- **Augmentation:** random crop (32x32 with 4-pixel padding) and random horizontal flip; inputs normalised with the CIFAR-10 channel mean and standard deviation.
- **Network:** three convolutional blocks (two 3x3 convolutions with batch normalisation and ReLU, max pooling, dropout 0.1 to 0.3), then a fully connected classifier; `forward()` returns softmax class probabilities.
- **Training:** cross-entropy loss (negative log-likelihood of the softmax output), SGD with Nesterov momentum and weight decay, one-cycle learning-rate schedule, 30 epochs, batch size 128.

## Files

| File | Contents |
|---|---|
| `CIFAR-10_Image_Classifier.ipynb` | Completed notebook with all outputs |
| `CIFAR-10_Image_Classifier.html` | HTML export of the notebook |
| `cifar10_cnn_weights.pth` | Trained model weights (`state_dict`), saved with `torch.save()` |

## Loading the trained model

```python
import torch
# define the Net class from the notebook first
model = Net()
model.load_state_dict(torch.load("cifar10_cnn_weights.pth", map_location="cpu"))
model.eval()
```
