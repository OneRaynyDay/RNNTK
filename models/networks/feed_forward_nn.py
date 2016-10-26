import numpy as np
from models.layers import *
from tools.nn_tools import *

""" 
A feedforward network is simply called a multilayered perceptron, or MLP.

It's basically only affine transformations followed by a softmax or SVM.
We'll use a softmax here.
"""

class VanillaRNN:
    def __init__(self, num_samples, input_dim, output_dim, hidden_sizes):
        """
        We allow the user to define how many affine layers he wants in the middle.
        If he wants none, the hidden_sizes would be []. This decomposes to simply an SVM/Softmax.
        
        If he wants 1 hidden layer, the hidden_sizes would be 
            [(input_dim, hidden_dim), (hidden_dim, output_dim)]
        Then we have the following:
            p("Wh")[0] and p("Wh")[1]
        
        """
        