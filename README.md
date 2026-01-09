# neural-network-jacobian-mnist
A NumPy based neural network train on MNIST, with visualizations of the Jacobian for individual examples.


# MNIST Neural Network Jacobian Visualization

## Programming Language
- **Python**

## Purpose of the Code
The purpose of this project is to provide a clearer understanding of which input features a neural network uses when classifying an image. A fully connected neural network is implemented from scratch using NumPy to maximize transparency and avoid reliance on high-level frameworks. The Jacobian of the network output with respect to the input is then computed and visualized as a heatmap overlaid on an example MNIST image, giving insight into which regions of the input most influence the model’s classification.

## Project Scale
- **Classes**
  - 1 class (`DeepNeuralNetwork`)
    - 1 constructor
    - 6 member functions
- **Helper functions**
  - 5 standalone helper functions
- **Total lines of code**
  -  181 lines

## Object-Oriented Programming Concepts
- Encapsulation
- Abstraction

## Data Structures Used
- NumPy arrays
- Python lists
- Tuples
- File objects
- Primitive data types (strings, integers, floats)

## Algorithms and Notable Implementations
This script uses matrix-based data structures (2D NumPy arrays) to compute the forward pass and backpropagation steps of a neural network. It implements the backpropagation algorithm with stochastic gradient descent to train the model, and then reuses backpropagation to compute the Jacobian of the network output with respect to the input. 

