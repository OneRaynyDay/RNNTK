import numpy as np
from models.layers import *
from tools.nn_tools import *
"""
Now that we have all the layers, we need to import them and kind of use them as
"lego blocks" for our application purpose.

A normal RNN looks like this:

[  ...  Ouput results  ...  ]
   |     |     |     |    |
[  ...  Softmax layer  ...  ]
   |     |     |     |    |
[  ...  Hidden layers  ...  ]
   |     |     |     |    |
[  ...  Word embedding ...  ]

So how do we make this?
Well, the basic pipeline is we need to pass in the word embedding vectors:

Constants:
    D = number of words
    V = size of each word vector (num of dimensions)
    N = number of samples
    T = number of time steps per sample

####### Inputs #######
    words_idx : (N,T)

####### Internals ########
    Words : (D,V)
    W_xh : (V,H)
    W_hh : (H,H)
    W_hy : (H,D)
    
What we need to do:

Constructor:
1. Randomly initialize Words, W_xh, W_hh, W_hy
Input words_idx.

Forward pass:
1. word_embedding_forward
2. rnn_step_forward
3. affine_forward
4. softmax_forward

Backward pass:
4. softmax_backward
3. affine_backward
2. rnn_step_backward
1. word_embedding_backward

"""

class VanillaRNN:
    def __init__(self, num_samples, num_words, time_dim, hidden_dim, word_vec_dim):
        """
        We assume words_idx has the shape (N,T)
        """
        self.constants = {}
        self.params = {}
        
        self.constants["N"] = num_samples
        self.constants["T"] = time_dim
        self.constants["H"] = hidden_dim
        self.constants["V"] = word_vec_dim
        self.constants["D"] = num_words
        
        # Save typing 
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        
        # words
        self.params["words"] = orthogonal_init((c("D"), c("V")))
        
        # W_xh should be a V x H matrix
        self.params["W_xh"] = orthogonal_init((c("V"), c("H")))
        
        # W_hh should be a H x H matrix
        self.params["W_hh"] = orthogonal_init((c("H"), c("H")))
        
        # W_hy should be a H x D matrix
        self.params["W_hy"] = orthogonal_init((c("H"), c("D")))
        
        # There are 2 biases: 1 for rnn and 1 for affine.
        self.params["b_rnn"] = orthogonal_init((c("H"),))
        self.params["b_affine"] = orthogonal_init((c("D"),))
        
    def loss(self, y, h0):
        """
        Note: To run the forward function, we need to have the weights set up. 
        
        This function returns all of its derivatives in a map, and the loss. 
        It does the forward and backwards pass in one call of the function.
        
        Inputs:
        y = (N,T)
        y is N samples of T time intervals of words. Each number in y represents
        an index inside of self.params["words"].
        h0 = (T,)
        """
        # Save typing 
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        
        # Step 1: Word embedding forward:
        x = word_embedding_forward(p("words"), y)
        
        # Step 2: rnn forward:
        h = rnn_forward(x, p("W_xh"), p("W_hh"), p("b_rnn"), h0)
        
        # Step 3: affine forward:
        affine = rnn_affine_forward(h, p("W_hy"), p("b_affine"))
        
        # step 4: softmax forward and backward:
        loss, dout = rnn_softmax(affine, y) 
        
        # step 3: affine backward:
        dh, dW_hy, db_affine = rnn_affine_backward(h, p("W_hy"), p("b_affine"), dout)

        # step 2: rnn backward:
        dW_hh, dW_xh, dx, db_rnn, dh0 = rnn_backward(x, p("W_xh"), p("W_hh"), p("b_rnn"), h0, h, dh)
        
        # step 1: Word embedding backward:
        dwords = word_embedding_backward(dx, p("words"), y)
        
        # Returns the loss and all the derivatives along with the fields.
        return (loss, [[p("words"), dwords], 
               [p("W_xh"), dW_xh], 
               [p("W_hh"), dW_hh],
               [p("W_hy"), dW_hy],
               [p("b_affine"), db_affine],
               [p("b_rnn"), db_rnn],
               [h0, dh0]])
                                              