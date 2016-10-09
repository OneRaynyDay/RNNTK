import numpy as np
from layers import *
from nn_tools import *
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
    def __init__(self, num_words, words_idx, hidden_dim, word_vec_dim):
        """
        We assume words_idx has the shape (N,T)
        """
        N,T = words_idx.shape
        self.constants["N"] = N
        self.constants["T"] = T
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
        
    def loss(self, x):
        """
        Note: To run the forward function, we need to have the weights set up. 
        
        This function returns all of its derivatives in a map, and the loss. 
        It does the forward and backwards pass in one call of the function.
        
        Inputs:
        x = (N,T)
        x is N samples of T time intervals of words. Each number in x represents
        an index inside of self.params["words"].
        """
        # Save typing 
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        
        # Step 1: Word embedding:
        word_matrix = word_embedding_forward(p("words"), x)
        
        # Step 2: 
        
    
                                              