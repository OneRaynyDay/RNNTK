import numpy as np

"""

Now that we have all the layers, we need to import them and kind of use them as
"lego blocks" for our application purpose.

A normal RNN looks like this:

[  ...  Ouput results  ...  ]
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
    x_idx : (N,T)
    x : (N,T,V)

####### Internals ########
    Words : (D,V)
    W_xh : (V,H)
    W_hh : (H,H)
    W_hy : (H,D)
    
What we need to do:

Constructor:
1. Randomly initialize Words, W_xh, W_hh, W_hy

Forward pass:
1. word_embedding_forward
2. rnn_step_forward
3. affine_forward

"""