import numpy as np
from models.layers import *
from tools.nn_tools import *
"""
Now that we have all the layers, we need to import them and kind of use them as
"lego blocks" for our application purpose.

A normal RNN looks like this:

[  ...  Ouput results  ...  ]
   |     |     |     |    |
[  ...  Softmax layer(dropout'd)  ...  ]
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
    def __init__(self, num_samples, num_words, time_dim, hidden_dim, word_vec_dim, l2_lambda = 0, dropout_keep_prob = 0.5):
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
        self.constants["p"] = dropout_keep_prob
        self.constants["l2_lambda"] = l2_lambda
        # Save typing 
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        
        func_init = variance_init
        
        # words
        self.params["words"] = func_init((c("D"), c("V")))
        
        # W_xh should be a V x H matrix
        self.params["W_xh"] = func_init((c("V"), c("H")))
        
        # W_hh should be a H x H matrix
        self.params["W_hh"] = func_init((c("H"), c("H")))
        
        # W_hy should be a H x D matrix
        self.params["W_hy"] = func_init((c("H"), c("D")))
        
        # There are 2 biases: 1 for rnn and 1 for affine.
        self.params["b_rnn"] = np.zeros((c("H"),))
        self.params["b_affine"] = np.zeros((c("D"),))
        
    def loss(self, x, y, h0):
        """
        Note: To run the forward function, we need to have the weights set up. 
        
        This function returns all of its derivatives in a map, and the loss. 
        It does the forward and backwards pass in one call of the function.
        
        Inputs:
        x = (N,T)
        y = (N,T)
        x and y are N samples of T time intervals of words. Each number in y represents
        an index inside of self.params["words"].
        h0 = (T,)
        """
        # Save typing 
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        
        # Step 1: Word embedding forward:
        words_chosen = word_embedding_forward(p("words"), x)
        
        # Step 2: rnn forward:
        h = rnn_forward(words_chosen, p("W_xh"), p("W_hh"), p("b_rnn"), h0)
        
        # Step 3: affine forward:
        affine = rnn_affine_forward(h, p("W_hy"), p("b_affine"))
        
        # step 4: softmax forward and backward:
        loss, dout = rnn_softmax(affine, y)
        
        # step 4.5: Add up the loss with the regularization parameters:
        # Question: Why do we not add the biases? It's because the bias is a discriminant
        # and it allows for the net to learn non-linear functions.
        # If we try to minimize the bias then we're limiting the expressiveness of the neural net.
        loss += c("l2_lambda")*(np.sum(p("W_xh")**2) + np.sum(p("W_hh")**2) + np.sum(p("W_hy")**2))/2
        
        # step 3: affine backward:
        dh, dW_hy, db_affine = rnn_affine_backward(h, p("W_hy"), p("b_affine"), dout)

        # step 2: rnn backward:
        dW_hh, dW_xh, dx, db_rnn, dh0 = rnn_backward(words_chosen, p("W_xh"), p("W_hh"), p("b_rnn"), h0, h, dh)
        
        # step 1: Word embedding backward:
        dwords = word_embedding_backward(dx, p("words"), x)
        
        # step 0.5: Add up the regularization losses
        dW_xh += c("l2_lambda")*(p("W_xh"))
        dW_hh += c("l2_lambda")*(p("W_hh"))
        dW_hy += c("l2_lambda")*(p("W_hy"))
        
        # Returns the loss and all the derivatives along with the fields.
        return (loss, [[p("words"), dwords], 
               [p("W_xh"), dW_xh], 
               [p("W_hh"), dW_hh],
               [p("W_hy"), dW_hy],
               [p("b_affine"), db_affine],
               [p("b_rnn"), db_rnn],
               [h0, dh0]], h[:,-1,:])

    def load(self, args):
        """
        Reloads the model into a previous state.
        
        """
        pass
    def predict(self, x, seq_len=6, h0=None):
        """
        Actually does the prediction using trained RNN.
        
        Input:
        x : a single word idx.
        
        Output:
        output : a numpy array of shape (seq_len,) that represents the sequence.
        """
        output = []
        c = lambda arg: self.constants[arg]
        p = lambda arg: self.params[arg]
        if h0 == None:
            prev_h = np.zeros((1, c("H")))
        else:
            prev_h = h0

        x = np.array([[x]])

        for i in xrange(seq_len):
            # Step 1: Word embedding forward:
            words_chosen = word_embedding_forward(p("words"), x)

            # Step 2: rnn forward:
            h = rnn_step_forward(prev_h, p("W_hh"), words_chosen, p("W_xh"), p("b_rnn"))

            # Step 3: affine forward:
            affine = rnn_affine_forward(h, p("W_hy"), p("b_affine"))

            # step 4: softmax forward:
            softmax_scores = np.exp(affine)/np.sum(np.exp(affine))

            x = np.random.choice(c("D"), p=softmax_scores.ravel())
            output.append(x)
            x = np.array([[x]])
            prev_h = h

        return output
        
    
                                              