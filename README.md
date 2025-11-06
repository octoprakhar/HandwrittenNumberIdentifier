# CNN from Scratch: Handwritten Number Classification (MNIST)

## Project Overview

This project implements a **Convolutional Neural Network (CNN) from scratch** in Python, using only `numpy` for core mathematical operations, to classify handwritten digits from the **MNIST dataset**.

The primary goal of this exercise was to gain a deep understanding of the fundamental building blocks of a CNN by manually implementing the **forward and backward passes** for all key layers, including convolution, ReLU activation, max pooling, flattening, and a final dense layer with softmax.

## Implementation Details

The core logic is contained within the `HandwrittenNumberClassification` class.

### Key Implemented Components:

* **Convolution Layer (`generate_feature_map`):** Implements 2D convolution using a sliding window technique. The input is a 4D array `(Batch, Height, Width, Channels)` and the kernel is `(Filters, Channels_in, Kernel_H, Kernel_W)`.
* **ReLU Activation (`relu`):** Element-wise application of the Rectified Linear Unit, $f(x) = \max(0, x)$.
* **Max Pooling Layer (`generate_max_pool`):** Implements a max pooling operation with a specified `pool_size` and `stride`. It also generates a **binary mask (`self.binary_max_pool`)** which is crucial for distributing gradients during backpropagation (unpooling).
* **Flattening (`generate_flattened_array`):** Reshapes the pooled feature maps into a 1D vector per sample for the dense layer.
* **Dense (Fully Connected) Layer:** Performs the final classification using a weights matrix (`self.W_dense`) and bias vector (`self.B_dense`).
* **Softmax (`softmax`):** Computes class probabilities for the final output.
* **Cross-Entropy Loss (`compute_loss`):** Calculates the loss for the classification task.
* **Backpropagation:**
    * **Decision Making (`backprop_decision_making`):** Calculates gradients for the dense layer weights/bias and propagates the loss back to the flattened array.
    * **Unpooling (`unpool`):** Redistributes the gradient from the pooled map back to the size of the pre-pooled feature map using the stored **binary mask**.
    * **ReLU Gradient:** Applies the derivative of ReLU (1 for positive inputs, 0 otherwise).
    * **Kernel Loss (`generate_kernal_loss`):** Calculates the gradient of the loss with respect to the convolution kernel weights, utilizing the input and the gradient from the ReLU layer.
* **Optimizer:** Uses **Stochastic Gradient Descent (SGD)** for weight updates based on the calculated gradients (`self.lr`).

## ⚙️ How to Run

### Prerequisites

* Python (3.x)
* `tensorflow` (only for loading the MNIST dataset)
* `numpy`

### Installation

```bash
pip install tensorflow numpy
