import numpy as np
from vanilla_rnn import *

"""
A solver model that intakes the rnn model and solves it according to the gradient
update function inputted as an argument.

The working pipeline goes something like this:

1. Solver intakes the model, passes in the necessary arguments.
2. Runs model's loss function over and over, collecting its gradients.
3. Uses its gradients and the associated gradient descent function to update weights.
4. After running for X epochs, it stops and returns its current model.
"""

class solver:
    def __init__(self, 
