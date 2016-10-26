import numpy as np

"""
This toolset is meant for use in a neural network. 

It currently features the following:

orthogonal_init(shape):
    Uses orthogonal random matrices as the starting initialization value for the 
    weights of the network. It works fairly well despite there being no research
    papers about it currently(except for Saxe et al.). 
    
TODO:
variance_init(shape):
"""

def orthogonal_init(shape, divisor=100):
    """
    Some intuition on why orthogonal initialization is good:
    For the fact that the matrix A*A' = I, we know that every vector
    dotted with every other vector yields 0 unless it is the same vector.
    
    In this case u has variable sized vectors but it spans into all subspaces equally, giving
    a good distribution of vectors to backpropagate on.
    
    References
    ----------
    .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
           "Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
    """
    W = np.random.randn(*shape)/divisor
    print W
    if len(W.shape) == 1:
        return W
    u, s, v = np.linalg.svd(W, full_matrices=False)
    if u.shape == shape: 
        return u 
    else: 
        return v.reshape(shape)

def variance_init(shape):
    """
    1/sqrt(fan_in)
    The shape is (In_out, Out_in), because (In_in, In_out) * (In_out, Out_in)
    """
    W = np.random.randn(*shape)/np.sqrt(shape[0])
    return W