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
```

## How to Run

1.  **Download or clone** the notebook file.
2.  **Open** the Jupyter Notebook: `HandwrittenNumberClassification.ipynb`.
3.  **Run all cells.** The notebook will:
    * Load and preprocess the MNIST data.
    * Initialize the `HandwrittenNumberClassification` model.
    * Train the model using a **small subset of the data** (`x_train[:100]`) for demonstration purposes, as the pure-numpy implementation is computationally intensive.

**Note:** For practical applications and full dataset training, high-level libraries like TensorFlow or PyTorch are required, but this project demonstrates the **internal mechanics** of these layers.

***

## Model Architecture (Self-Defined Parameters)

The model is configured with the following default parameters, defined in the `__init__` method:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `batch_size` | 32 | Number of samples processed per batch. |
| `channel_size` | 1 | Input channels (Grayscale). |
| `kernal_size` | 4 | Convolutional kernel (filter) size (4x4). |
| `filters` | 8 | Number of output feature maps. |
| `stride` | 4 | Stride for the max-pooling layer. |
| `pool_size` | 4 | Size of the max-pooling window (4x4). |
| `lr` | 0.01 | Learning rate for weight updates. |

***

## Learning & Takeaways

Implementing this CNN from the ground up provided invaluable insight into:

* **Dimensionality Management:** Understanding the complex changes in array shapes (e.g., $(B, H, W, C)$ to $(B, F, L_H, L_W)$ to $(B, F \times \dots)$) across different layers.
* **Gradient Flow:** Precisely how the loss is backpropagated and distributed through pooling and convolutional layers. The **binary mask** in max-pooling is critical for sparse gradient distribution.
* **Matrix Operations:** Understanding the role of `np.sum`, broadcasting, and element-wise multiplication in calculating gradients for the weights (kernels).
