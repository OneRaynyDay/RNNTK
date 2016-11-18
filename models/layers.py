import numpy as np

"""

This file contains all the good shit: the layers for our recurrent neural networks. As required,
we also have affine and word_embedding layers(but are not a part of our recurrent NN) also included.

"""

def sigmoid(x):
    """
    Numerically-stable sigmoid function.
    
    Recall that sigmoid is equal to:
    1/(1+exp(-x))
    
    However, if z is negative then we will have gigantic values which
    may not be safe for floating points. No explosions, please!
    
    Recall that if we multiply both numerator and denominator by exp(z):
    (1*exp(x))/((1+exp(-x)*exp(x))
    
    We get:
    exp(x)/(exp(x) + 1)
    
    This is very numerically stable as all the numbers are guarranteed to be
    less than 1. GOOGLE PLS HIRE ME 
    """
    z = x.copy()
    z[ x >= 0 ] = 1 / (1 + np.exp(-z[ x >= 0 ]))
    z[ x < 0 ] = np.exp(z[ x < 0 ]) / (1 + np.exp(z[ x < 0 ]))
    return z

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
    words_dim = (D,V)
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
    dout = dout*(1-(np.tanh(prev_h.dot(W_hh) + x.dot(W_xh) + b)**2))
    dW_hh = (prev_h.T).dot(dout)
    dW_xh = (x.T).dot(dout)
    dprev_h = dout.dot(W_hh.T)
    dx = dout.dot(W_xh.T)
    db = np.sum(dout, axis = 0)
    
    return dW_hh, dW_xh, dprev_h, dx, db

def lstm_step_forward(prev_h, W_hh, x, W_xh, b, prev_c):
    """
    LSTM cells are much more complicated than vanilla RNN cells.
    LSTM cells have 4 properties:
    i, f, o, g.
    They're acquired via taking sigmoid() and tanh() of 
    [W_ixh, W_ihh], [W_fxh, W_fhh], ...
    With [x, prev_h].
    
    In our inputs, we have 
    X = (N,V)
    prev_h = (N,H)
    prev_c = (N,H)
    W_xh = (V,4H)
    W_hh = (H,4H)
    b = (4H,)
    """
    _, H = prev_h.shape
    cache = {} # Too many fkin values dude
    
    dot_prod = x.dot(W_xh) + prev_h.dot(W_hh) + b # Gives us N,4H
    
    # To get i,f,o,g, we need to extract out the 
    # individual columns and apply transformations:
    i = sigmoid(dot_prod[:,:H])
    f = sigmoid(dot_prod[:,H:H*2])
    o = sigmoid(dot_prod[:,H*2:H*3])
    g = np.tanh(dot_prod[:,H*3:])
    
    # Now, we need to use these values to get c and h.
    c = f * prev_c + (i * g)
    h = np.tanh(c) * o
    
    cache["i"] = i
    cache["f"] = f
    cache["o"] = o
    cache["g"] = g
    cache["c"] = c
    cache["prev_c"] = prev_c
    cache["prev_h"] = prev_h
    
    return cache, c, h

def lstm_step_backward(W_hh, x, W_xh, b, cache, dh, dc):
    """
    LSTM cells have shitty backprop - they're really long. 
    We brought back the cache so we don't have to compute the values again.
    
    This is subject to optimization, obviously, but as Donald Knuth or some1 said:
    "Premature optimization is the root of all evil".
    """
    _, H = cache["prev_h"].shape
    
    # We should start at dh - that one's longer
    _1 = dh*cache["o"]
    _2 = _1*(1-np.tanh(cache["c"])**2)
    _3 = _2 + dc
    _4 = _3
    _5 = _3
    _6 = _5*cache["i"]
    _7 = _6*(1-cache["g"]**2) # g gate
    _8 = _5 * cache["g"]
    _9 = _8 * cache["i"] * (1-cache["i"]) # i gate
    _10 = _4 * cache["prev_c"]
    _11 = _10 * cache["f"] * (1-cache["f"]) # f gate
    dprev_c = _4 * cache["f"] # This is also _12.
    _13 = dh*np.tanh(cache["c"])
    _14 = _13 * cache["o"] * (1-cache["o"]) # o gate
    
    # Because we're going to be doing the same tiling process for each gate,
    # might as well tile them all together right now.
    tiled = np.concatenate([_9, _11, _14, _7], axis=1)
        
    # Now that we have the LSTM structure backprop done, we need to 
    # aggregate them inside of db, dW_hh, dW_xh, dprev_h and dx.
    db = np.sum(tiled, axis=0)
        
    # If dW_hh is in the shape H,4H, then we will take each slice of H,xH:yH
    # to be a slice of the specific gate. 
    # prev_h is in the shape of N,H. prev_h.T.dot(derivative) 
    # dW_hh = prev_h.T.dot(_9
    dW_hh = cache["prev_h"].T.dot(tiled)

    # If dW_xh is in the shape V,4H, then we will take each slice of V,xH:yH
    # to be a slice of the specific gate.
    # x is in the shape of N,V. x.T.dot(derivative)
    dW_xh = x.T.dot(tiled)
    
    # dx is in the shape of N,V, and we use dx in every single gate.
    # W_xh is in the shape of V,H. derivative.dot(W_xh.T) 
    dx = tiled.dot(W_xh.T)
    
    # dprev_h is in the shape of N,h, and we use dprev_h in every single gate.
    # W_hh is in the shape of H,H. derivative.dot(W_hh.T)
    dprev_h = tiled.dot(W_hh.T)
    
    return dW_hh, dW_xh, dprev_h, dx, db, dprev_c
    
    
def rnn_forward(x, W_xh, W_hh, b, h0=None):
    """
    Given the following parameters: 
    h0 = (N,H)
    W_hh = (H,H)
    x = (N,T,V)
    W_xh = (D,H)
    b = (H,)
    
    We make h = (N,T,H).
    
    Note: We make the initial state h0 all zeros.
    We can definitely change this if we wanted to.
    This is used to continuously give sequences.
    
    Run the entire hidden state layer so we can get h.
    returns h = (N,T,H)
    """
    N,T,V = x.shape
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
    x = (N,T,V)
    W_xh = (D,H)
    b = (H,)
    h = (N,T,H)
    dout = (N,T,H)
    
    dW_hh, dW_xh, dh[:,i-1,:], dx, db += rnn_step_backward(prev_h, W_hh, x, W_xh, b, dout)
    
    """
    N,T,V = x.shape
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
        
def lstm_forward(x, W_xh, W_hh, b, h0=None):
    """
    Given the following parameters: 
    h0 = (N,H)
    W_hh = (H,4H)
    x = (N,T,D)
    W_xh = (D,4H)
    b = (H,)
    
    We make h = (N,T,H).
    
    Note: We can make the initial state h0 all zeros.
    We can definitely change this if we wanted to.
    This is used to continuously give sequences.
    
    We also pass in the value c0 = 0. This is a hidden vector that's
    not exposed. It's always the zero vector to start with, although 
    I'm not sure what the effect of not passing back past epochs 
    really is on the result.
    
    Run the entire hidden state layer so we can get h.
    returns h = (N,T,H)
    """
    N,T,V = x.shape
    H,_ = W_hh.shape
    h, c = np.zeros((N,T,H)), np.zeros((N,T,H))
    caches = []
    c0 = np.zeros((N,H))
    if h0 == None: # Supply an h0 state.
        h0 = np.zeros((N,H))    
    
    for i in xrange(T):
        if i == 0:
            cache, c[:,i,:], h[:,i,:] = lstm_step_forward(h0, W_hh, x[:,i,:], W_xh, b, c0)
        else:
            cache, c[:,i,:], h[:,i,:] = lstm_step_forward(h[:,i-1,:], W_hh, x[:,i,:], W_xh, b, c[:,i-1,:])
        caches.append(cache)
    return caches, h
    
def lstm_backward(x, W_xh, W_hh, b, h0, caches, dout):
    """
    Given the following parameters:
    W_xh = (H,H)
    x = (N,T,V)
    W_xh = (D,H)
    b = (H,)
    h = (N,T,H)
    dout = (N,T,H)
    
    dW_hh, dW_xh, dh[:,i-1,:], dx, db += rnn_step_backward(prev_h, W_hh, x, W_xh, b, dout)
    """

    N,T,V = x.shape
    H,_ = W_hh.shape
    dW_hh = np.zeros_like(W_hh)
    dW_xh = np.zeros_like(W_xh)
    #dh = dout
    dc = np.zeros_like(dout)
    dx = np.zeros_like(x)
    db = np.zeros_like(b)
    dprev_h = np.zeros((N,H))
    
    # rnn_step_backward args: W_hh, x, W_xh, b, cache, dh, dc 
    for i in reversed(xrange(T)):
        # Because python does not allow multiple += assignments via unpacked tuples
        # at the same time, we have to assign temporary values in order for this
        # to work properly.
        _dW_hh, _dW_xh, dprev_h, _dx, _db, _dc = lstm_step_backward(W_hh, 
                                                                x[:,i,:], 
                                                                W_xh, 
                                                                b, 
                                                                caches[i], 
                                                                dout[:,i,:]+dprev_h, 
                                                                dc[:,i,:])
        dc[:,i-1,:] = _dc
        dW_hh += _dW_hh
        dW_xh += _dW_xh
        dx[:,i,:] += _dx
        db += _db
    
    dh0 = dprev_h
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

def dropout_forward(x, p = 0.5, mode="train"):
    """
    Dropout is a popular regularization technique founded by Geoff Hinton.
    Its idea is simple: just remove some cells' activation layers.
    Set them to 0 so that they don't contribute to the overall architecture.
    
    This greatly regularizes the nodes and when we want to get the testing result,
    we just simply multiply each activation by p to allow all of them to be still alive.
    
    Given implementation is based on the idea of "inverted dropout", which
    dropouts and scales in the training process by itself. 
    
    Input:
    x = (N,T,D) in a recurrent neural network. (N,D) otherwise
    p = probability of dropout. Between 0 and 1.
    mode = whether we are running the training version or the testing version.
    
    returns:
    mask = (N,T,D) of 0's and 1's, signifying which ones were dropped out. {ONLY during training}
    x = (N,T,D)
    """
    if mode == "train":
        mask = np.array(np.random.random(x.shape) < p, float)
        mask /= p # divide by p for the inverted dropout effect.
        
        x *= mask
        return x, mask
    elif mode == "test":
        # Note how we didn't have to divide by p this time due to the inverted dropout
        return x

def dropout_backward(mask, dout, mode="train"):
    """
    Backpropagation of dropout is fairly intuitive. If we set the value to 0, then
    there is literally no gradient passing back. We "snip" them off of the gradients per se.
    
    Input:
    x = (N,T,D) in a recurrent neural network.
    """
    if mode == "train":
        dout *= mask
        return dout
    elif mode == "test":
        return dout

def rnn_affine_forward(h, W_hy, b):
    """
    Given the following parameters: h, W_hy, b
    rnn_affine_rdforwa takes care of rnn situations where there is
    a temporal argument:
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    h = (N,T,H)
    W_hy = (H,D)
    b = (D,)
    
    returns (N,T,H)
    Note: In this case we just happened to have the correct answer
    by using affine_forward. Only do this for minimum code-rewriting!
    This might not work 100% of the time.
    """
    return affine_forward(h, W_hy, b)
    
def rnn_affine_backward(h, W_hy, b, dout):
    """
    Given the following parameters: h, W_hy, b, dout
    rnn_affine_backward takes care of rnn situations where there is
    a temporal argument:
    
    Constants:
    D = number of unique words in our corpus
    N = number of samples
    H = the dimension of hidden cell vector
    
    h = (N,T,H)
    W_hy = (H,D)
    b = (D,)
    dout = (N,T,D)
    
    **************************************************
    Note: We see that, for general rule of thumb, 
    A*B = C where A,B,C are matrices:
    
    dA = dC*B.T
    dB = A.T*dC
    **************************************************
    
    dh = dout * W_hy.T = (N,T,D) * (D,H) = (N,T,H)
    dW_hy = h.T * dout = (H,T*N) * (N*T,D) = (H,D)
    db = sum(axis=0, 1) of dout
    """
    N,T,H = h.shape
    dh = dout.dot(W_hy.T)
    dW_hy = (h.reshape(N*T,H).T).dot(dout.reshape(N*T,-1))
    db = np.sum(dout, axis = (0,1))
    
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
    x -= np.max(x, axis=1, keepdims=True) # subtraction for numeric stability
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
    dJ1 = np.ones((N,D), float)/(denominator[:, np.newaxis])
    dJ2 = np.array(y_mask, float)/(numerator[:, np.newaxis])
    dJ = (dJ1 - dJ2)*x_exp/N    
    
    return loss, dJ

def rnn_softmax(x, y):
    """
    Softmax is a probabilistic cost function, whereas the domain of the function gets squashed
    to [0,1] and the cost is restrained from [0, inf).
    
    This is a special rnn_softmax function where it calculates the softmax score of each temporal score.
    This changes the x dim from:
    (N,T,D) to (N*T,D) 
    (Because we need to keep the one-hot cost area the same dimension, but need to squash down to dim=2)
    Then, we calculate each score using the normal softmax function, then wrap it up back to the correct shape.
    """
    
    N,T,D = x.shape
    x = x.reshape(N*T,D) # Change it to 2d array for softmax
    y = y.ravel() # Change it to a 1d array for softmax
    loss, dJ = softmax(x, y)
    loss *= T # We actually divided the loss by (N*T) inside, so we multiply by T outside.
    dJ *= T # We actually divided the dJ the same way.
    return loss, dJ.reshape(N,T,D)


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