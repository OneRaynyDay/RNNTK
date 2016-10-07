import numpy as np

"""

This file contains all the good shit: the layers for our recurrent neural networks. As required,
we also have affine and word_embedding layers(but are not a part of our recurrent NN) also included.

"""

def word_embedding_forward(words, x):
    """ 
    Given the following parameters: Words, x
    
    Constants:
    D = number of words
    V = size of each word vector (num of dimensions)
    N = number of samples
    T = number of time steps per sample
    
    words = ndarray. Shape = (D,V). 
    x = ndarray. Shape = (N,T)
    
    E.x.:
    >>> words = np.array([[12,3,5],[3,66,2],[1,2,3],[0,5,0]])
    >>> x = np.array([3,2,1,2,3])
    >>> words[x]
    array([[ 0,  5,  0],
           [ 1,  2,  3],
           [ 3, 66,  2],
           [ 1,  2,  3],
           [ 0,  5,  0]])
    
    We are to construct a word embedding vector for the currently selected words. 
    """
    return words[x] # This is a N,T,V matrix. 
    # We took a 1xV vector and put it where the index of x used to be.
    # x[i,j] = words[x[i,j]]
    
def word_embedding_backward(dout, words_dim, x):
    """ 
    Given the following parameters: Words, x, dout
    
    Constants:
    D = number of words
    V = size of each word vector (num of dimensions)
    N = number of samples
    T = number of time steps per sample
    
    dout = (N, T, V)
    x = ndarray. Shape = (N,T)
    words_dim = tuple(D,V)
    dWords = (D,V)
    We know that the derivative is 1 whenever we just choose the vector at index x[i,j].
    To go backwards, we can say that the specific i-th vector's derivative at dout[x] 
    is going to be put in dWords[x[i,j]].
    
    returns a (D,V) to the Words matrix.
    """
    
    dW = np.zeros(words_dim)
    np.add.at(dW, x, dout)
    return dW

def rnn_step_forward(prev_h, W_hh, x, W_xh, b):
    """
    Given the following parameters: prev_h, W_hh, x, W_xh, b
    
    Constants:
    V = size of each word vector (num of dimensions)
    N = number of samples
    H = the dimension of hidden cell vector
    
    prev_h = (N,H)
    x = (N, V)
    W_hh = (H,H)
    W_xh = (V,H)
    b = (H,)
    
    The equation for a vanilla RNN single step is this:
    h = tanh(prev_h * W_hh + x * W_xh + b)
    
    We call it a single step because it only does 1 input at a time.
    In this case, we could be only computing h_0, when really the hidden layer looks like:
    
         y_0 y_1 y_2 ... y_n
          |   |   |       |
    h_-1-h_0-h_1-h_2-...-h_n
          |   |   |       |
         x_0 x_1 x_2 ... x_n
    
    returns a (N,H). If we received h_0 and x_1, then we would output h_1. 
    """
    
    return np.tanh(prev_h.dot(W_hh) + x.dot(W_xh) + b)

def rnn_step_backward(prev_h, W_hh, x, W_xh, b, dout):
    """
    Given the following parameters: prev_h, W_hh, x, W_xh, b, dout
    
    Constants:
    V = size of each word vector (num of dimensions)
    N = number of samples
    H = the dimension of hidden cell vector
    
    prev_h = (N,H)
    x = (N, V)
    W_hh = (H,H)
    W_xh = (V,H)
    b = (H,)
    dout = (N,H)
    
    derivative of tanh' is 1-tanh^2.
    
    dW_hh is (H,H).
    (H,H) = (dout.T * prev_h).T = (H,N) * (N,H) = (H,H)
          = prev_h.T * b = (H,N) * (N,H).
    
    dW_xh is (V,H).
    (V,H) = dout.T * x = ((H,N) * (N,V)).T = (H,V).T = (V,H)
          = x.T * dout = (V,N) * (N,H) = (V,H)
    
    dprev_h is (N,H).
    (N,H) = dout * W_hh.T = (N,H) * (H,H) = (N,H)
    
    dx is (N, V).
    (N,V) = dout * W_xh.T = (N,H) * (H,V) = (N,V)
    
    db is (H,).
    (H,) = sum(axis = 0) of dout = (H,)
    """
    dout = 1-np.tanh(prev_h.dot(W_hh) + x.dot(W_xh) + b)**2
    dW_hh = (prev_h.T).dot(dout)
    dW_xh = (x.T).dot(dout)
    dprev_h = dout.dot(W_hh.T)
    dx = dout.dot(W_xh.T)
    db = np.sum(dout, axis = 0)
    
    print dW_hh.shape, dW_xh.shape, dprev_h.shape, dx.shape, db.shape
    return dW_hh, dW_xh, dprev_h, dx, db
    
def affine_forward(h, W_hy, b):
    """
    Given the following parameters: h, W_hy, b
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    h = (N,H)
    W_hy = (H,D)
    b = (D,)
    
    This is the good part. We're going to do an affine_forward to 
    get our hidden states into actual word vectors. It is essentially
    a large one-hot vector that represents a specific word in the D 
    number of words we have in the map.
    
    We need an affine_forward and not just any ordinary transform/skew/rotate,
    because we want to drag the vector off the origin for more variety.
    
    Returns (N,D)
    """
    return h.dot(W_hy) + b

def affine_backward(h, W_hy, b, dout):
    """
    Given the following parameters: h, W_hy, b, dout
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    h = (N,H)
    W_hy = (H,D)
    b = (D,)
    dout = (N,D)
    
    **************************************************
    Note: We see that, for general rule of thumb, 
    A*B = C where A,B,C are matrices:
    
    dA = dC*B.T
    dB = A.T*dC
    **************************************************
    
    dh = dout * W_hy.T = (N,D) * (D,H) = (N,H)
    dW_hy = h.T * dout = (H,N) * (N,D) = (H,D)
    db = sum(axis=0) of dout
    """
    dh = dout.dot(W_hy.T)
    dW_hy = (h.T).dot(dout)
    db = np.sum(dout, axis = 0)
    
    return dh, dW_hy, db
    
    