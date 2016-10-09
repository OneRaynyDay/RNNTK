import numpy as np

"""
This toolset is meant for use in a neural network. 

It currently features the following:

orthogonal_init(shape):
    Uses orthogonal random matrices as the starting initialization value for the 
    weights of the network. It works fairly well despite there being no research
    papers about it currently(except for Saxe et al.). 
"""

def orthogonal_init(shape):
    """
    Some intuition on why orthogonal initialization is good:
    For the fact that the matrix A*A' = I, we know that every vector
    dotted with every other vector yields 0 unless it is the same vector.
    
    In this case u has variable sized vectors but it spans into all subspaces equally, giving
    a good distribution of vectors to backpropagate on.
    """
    W = np.random.randn(*shape)
    u, s, v = np.linalg.svd(W)
    return u