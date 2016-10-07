import numpy as np


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