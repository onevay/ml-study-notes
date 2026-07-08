# Perceptron and Adaline: From Idea to Gradient Descent

## Introduction
A brief overview of the topic and its significance in machine learning. This chapter covers the foundational algorithms that paved the way for modern neural networks, introducing key concepts such as weight updates, activation functions, and gradient-based optimization.

## Perceptron
### Mathematics
The perceptron computes a weighted sum of inputs and applies a step function to produce a binary output:

$$z = w_1x_1 + w_2x_2 + \dots + w_mx_m + b = \mathbf{w}^T\mathbf{x} + b$$

$$\phi(z) = \begin{cases} 1, & \text{if } z \ge 0 \\ -1, & \text{if } z < 0 \end{cases}$$

### Learning Algorithm
1. Initialize weights (typically to zero or small random values).
2. For each epoch (full pass over the training data):
   - For each training sample $(x^{(i)}, y^{(i)})$:
     a. Compute prediction: $\hat{y}^{(i)} = \phi(z^{(i)})$
     b. Compute error: $error = y^{(i)} - \hat{y}^{(i)}$
     c. Update weights: $\Delta w_j = \eta \cdot error \cdot x_j^{(i)}$
     d. Update bias: $\Delta b = \eta \cdot error$
3. Repeat until all samples are correctly classified or the maximum number of epochs is reached.

### Limitations
- **Linear Separability**: The perceptron only converges if the data is linearly separable. For non-linearly separable data (e.g., XOR problem), the algorithm will never converge.
- **Binary Output**: The step function produces discrete outputs, making it unsuitable for probabilistic predictions.
- **No Confidence Measure**: The perceptron provides no information about the confidence of its predictions.

## Adaline (Adaptive Linear Neuron)
Adaline improves upon the perceptron by using a continuous activation function during training, enabling gradient-based optimization.

### Loss Function: Mean Squared Error (MSE)
$$L(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^{n} \big( y^{(i)} - z^{(i)} \big)^2$$

where $z^{(i)} = \mathbf{w}^T\mathbf{x}^{(i)} + b$ is the linear output.

### Gradient Descent
The weights are updated by moving in the direction opposite to the gradient of the loss function:

$$\mathbf{w} := \mathbf{w} - \eta \cdot \nabla L(\mathbf{w})$$

where $\nabla L(\mathbf{w})$ is the gradient vector of partial derivatives with respect to each weight.

**Key Advantage**: Adaline's loss function is differentiable, making it compatible with gradient descent and enabling efficient optimization.

## Comparison and Conclusions

| Feature | Perceptron | Adaline |
|---------|------------|---------|
| **Activation Function** | Step (non-differentiable) | Linear (during training) / Step (for prediction) |
| **Learning Rule** | Based on classification error | Based on continuous error (MSE) |
| **Convergence** | Only for linearly separable data | Converges for most data with proper learning rate |
| **Optimization** | No gradient used | Uses gradient descent |
| **Probabilistic Output** | No | No (but can be extended to logistic regression) |

**Key Takeaway**: Adaline bridges the gap between the perceptron and modern neural networks by introducing differentiable loss functions and gradient-based optimization—cornerstone techniques that underpin deep learning today.
