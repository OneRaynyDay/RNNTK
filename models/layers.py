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
    
def word_embedding_backward(dout, words, x):
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
    words_dim = words.shape
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
    
    return dW_hh, dW_xh, dprev_h, dx, db

def rnn_forward(x, W_xh, W_hh, b, h0=None):
    """
    Given the following parameters: 
    h0 = (N,H)
    W_hh = (H,H)
    x = (N,T,D)
    W_xh = (D,H)
    b = (H,)
    
    We make h = (N,T,H).
    
    Note: We make the initial state h0 all zeros.
    We can definitely change this if we wanted to.
    This is used to continuously give sequences.
    
    Run the entire hidden state layer so we can get h.
    returns h = (N,T,H)
    """
    N,T,D = x.shape
    H,_ = W_hh.shape
    h = np.zeros((N,T,H))
    if h0 != None: # Supply an h0 state.
        h[:,-1,:] = h0
    
    for i in xrange(T):
        h[:,i,:] = rnn_step_forward(h[:,i-1,:], W_hh, x[:,i,:], W_xh, b)
    return h

def rnn_backward(x, W_xh, W_hh, b, h0, h, dout):
    """
    Given the following parameters:
    W_xh = (H,H)
    x = (N,T,D)
    W_xh = (D,H)
    b = (H,)
    h = (N,T,H)
    dout = (N,T,H)
    
    dW_hh, dW_xh, dh[:,i-1,:], dx, db += rnn_step_backward(prev_h, W_hh, x, W_xh, b, dout)
    
    """
    N,T,D = x.shape
    H,_ = W_hh.shape
    dW_hh = np.zeros_like(W_hh)
    dW_xh = np.zeros_like(W_xh)
    dh = dout
    dx = np.zeros_like(x)
    db = np.zeros_like(b)
    h[:,-1,:] = h0
    
    # rnn_step_backward args: prev_h, W_hh, x, W_xh, b, dout
    for i in reversed(xrange(T)):
        # Because python does not allow multiple += assignments via unpacked tuples
        # at the same time, we have to assign temporary values in order for this
        # to work properly.
        _dW_hh, _dW_xh, _dh, _dx, _db = rnn_step_backward(h[:,i-1,:], W_hh, x[:,i,:], W_xh, b, dh[:,i,:])
        if i - 1 >= 0:
            dh[:,i-1,:] += _dh
        else:
            dh0 = _dh
        dW_hh += _dW_hh
        dW_xh += _dW_xh
        dx[:,i,:] += _dx
        db += _db

    return dW_hh, dW_xh, dx, db, dh0
        
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

def softmax(x, y):
    """
    Softmax is a probabilistic cost function, whereas the domain of the function gets squashed
    to [0,1] and the cost is restrained from [0, inf).
    We are assuming a log probability given a specific event(x_yi) happening given all other events.
    The other events should be small, while the chosen event(the one with the correct label) is high.
    
    Given the following parameters: x, y
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    # In the case of the rnn #
    x = (N,D)
    y = (N,) # the idx for the correct word.
    y can be used to get the correct indices: [np.range(1,n), y] = 1 => (N,D)
    
    returns: single number(which is the cost of the current function) + the derivative.
    
    ~~~For LOSS~~~
    Step 1. Affine transform (in another layer):
    affine = x * W_x + b = (N,D)
    
    Step 3. Perform the exponent operation on affine:
    affine = np.exp(affine)
    numerator = affine[np.range(1,n), y]
    denominator = np.sum(affine)
    
    Step 4. Perform the negative log distribution function on exponent result
    loss = -np.log(numerator/denominator)
    
    Step 5. If we have multiple samples, the result is the average of all scores in each D dimension.
    loss = 1/N*np.sum(loss)
    
    The derivative must be computed by hand ;)
    """
    
    N,D = x.shape

    ### For LOSS ###
    y_mask = np.zeros_like(x, int)
    y_mask[np.arange(N), y] = 1
    x_exp = np.exp(x)
    numerator = np.sum(x_exp * y_mask, axis=1)
    denominator = np.sum(x_exp, axis=1)
    softmax = -1*np.log(numerator/denominator)
    loss = np.sum(softmax)/N
    
    ### For dx ###
    dJ = 1 # Every derivative backflow starts at 1
    
    # The derivative splits here, so we have to call upstream numerator derivative dJ1,
    # and the downstream denominator derivative dJ2
    dJ1 = np.ones((N,D), float)/(denominator[:, np.newaxis] * N)
    dJ2 = np.array(y_mask, float)/(numerator[:, np.newaxis] * N)
    dJ = (dJ1 - dJ2)*x_exp    
    
    return loss, dJ

def SVM(x, y):
    """
    SVM, or Support Vector Machine is a linear model that tries to separate
    the score of the correct label from the rest with the optimized DELTA space
    between them. If they are outside of the DELTA gap, then the loss should be 0.
    Otherwise, it is a linear loss. (We also call this a hinge loss)
    
    Given the following parameters: x, y
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    # In the case of the rnn #
    x = (N,D)
    y = (N,) # the idx for the correct word.
    y can be used to get the correct indices: [np.range(1,n), y] = 1 => (N,D)
    
    returns: single number(which is the cost of the current function) + the derivative.
    
    ~~~For LOSS~~~
    Step 1. affine transform: refer to softmax
    
    Step 2. Subtract all vectors on that row with the correct labels score and add a delta.
    correct_score = affine[np.range(1,n), y]
    affine -= correct_score
    mask = np.ones_like(affine)
    mask[np.range(1,n), y] = 0
    affine += mask
    
    Step 3. The scores that are negative(if correct score > (incorrect score + DELTA)) are hinged to 0
    loss = max(affine, 0)
    
    Step 4. 
    If we have multiple samples, the result is the average of all scores in each D dimension.
    loss = 1/N*np.sum(loss)
    
    The derivative must be computed by hand ;)
    """
    N,D = x.shape
    

    ### For LOSS ###
    correct_scores = x[np.arange(N), y]
    x -= correct_scores[:, np.newaxis]
    mask = np.ones_like(x)
    mask[np.arange(N), y] = 0
    x += mask
    
    loss = np.maximum(x, 0)
    loss = np.sum(loss) / N
    
    ### For dx ###
    dJ = 1 # Every derivative backflow starts at 1
    dJ_mask = np.ones((N,D), float)
    dJ_mask[np.where(x < 0)] = 0
    dJ = dJ_mask / N * dJ
    dJ += (-1 * (np.ones_like(x) - mask) * D * dJ)
    
    return loss, dJ