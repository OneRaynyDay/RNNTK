import numpy as np

"""
A solver model that intakes the rnn model and solves it according to the gradient
update function inputted as an argument.

The working pipeline goes something like this:

1. Solver intakes the model, passes in the necessary arguments.
2. Runs model's loss function over and over, collecting its gradients.
3. Uses its gradients and the associated gradient descent function to update weights.
4. After running for X epochs, it stops and returns its current model.
"""
def adagrad(self, l):
    """ 
        Adagrad is an adaptive learning method. Examine the code below - it's fairly straight forward.
        It allows for robust choosing of learning_rates.
    """
    if self.cache == None:
        self.cache = []
        [self.cache.append(np.zeros_like(pair[0])) for pair in l]
    for i, pair in enumerate(l):
        self.cache[i] += pair[1]**2
        pair[0] += -1*self.learning_rate*pair[1]/(np.sqrt(self.cache[i]) + 1e-7) # 1e-7 to prevent NaN
        
def rmsprop(self, l):
    """
        RMSProp is basically leaky adagrad.
    """
    if self.cache == None:
        self.cache = []
        [self.cache.append(np.zeros_like(pair[0])) for pair in l]
    for i, pair in enumerate(l):
        self.cache[i] = self.decay_rate * self.cache[i] + (1.0-self.decay_rate) * pair[1]**2
        pair[0] += -1*self.learning_rate*pair[1]/(np.sqrt(self.cache[i]) + 1e-7) # 1e-7 to prevent NaN
        
def adam(self, l):
    """
        adam is probably the best gradient solver in all of these selections.
        It's a combination of RMSprop and momentum.
        However, it's got a shit ton of configurations:
        learning_rate - standard
        beta1 - beta1 is for the momentum update
        beta2 - beta2 is for the decay rate
        m - m is the momentum
        v - v is the rmsprop cache
    """
    if self.m == None and self.v == None:
        self.v = []
        self.m = []
        [self.v.append(np.zeros_like(pair[0])) for pair in l]
        [self.m.append(np.zeros_like(pair[0])) for pair in l]
    for i, pair in enumerate(l):
        self.m[i] = self.beta1 * self.m[i] + (1.0-self.beta1)*pair[1]
        self.v[i] = self.beta2 * self.v[i] + (1.0-self.beta2)*(pair[1]**2)
        pair[0] += -1*self.learning_rate * self.m[i]/(np.sqrt(self.v[i]) + 1e-7) # 1e-7 to prevent NaN

def momentum(self, l):
    """ 
        Momentum is a higher order gradient solver which caches the previous
        "velocity" and adds the current vector to the previous velocity vector.
        
        A better implementation of momentum is called nesterov momentum. We don't implement
        this because it fits awkwardly with our api interface.
    """
    if self.v == None:
        self.v = []
        [self.v.append(np.zeros_like(pair[0])) for pair in l]
    for i, pair in enumerate(l):
        self.v[i] = self.mu*self.v[i] - self.learning_rate*pair[1]
        pair[0] += self.v[i]

def sgd(self, l):
    """ 
        SGD is one of the classic gradient solving techniques.
        It's simply going down the slope using a constant learning_rate.
        In reality it's not very good, so we only use this for sanity checks.
    """
    for pair in l:
        pair[0] -= pair[1]*self.learning_rate

class Solver:
    def __init__(self, configs):
        """ Inputs:
        nn = the neural net object. We assume it has a loss() function that returns the following:
            1. the loss
            2. the gradients in the form of [(x1, dx1), (x2, dx2), ... , (xn, dxn)]
            3. any extra arguments afterwards(in the rnn's case, it's the hidden layer cell)
        
        config = a list of configurations for learning:
            1. learning_rate : number
            2. type : a string, in the set {"sgd", "momentum", "nesterov", "adagrad", "RMSprop", "adam"}
                Tip: so far, adam has been performing the best. So use that one.
            3. any extra variables for the specific type of upgrade.
        """
        self.update_map = {"sgd":sgd, 
                  "momentum":momentum, 
                  "adagrad":adagrad, 
                  "rmsprop":rmsprop, 
                  "adam":adam}
        self.learning_rate = configs["learning_rate"]
        self.func = self.update_map[configs["type"]]
        if configs["type"] == "momentum":
            self.mu = configs["mu"]
            self.v = None
        if configs["type"] == "adagrad":
            self.cache = None
        if configs["type"] == "rmsprop":
            self.cache = None
            self.decay_rate = configs["decay_rate"]
        if configs["type"] == "adam":
            self.m = None
            self.v = None
            self.beta1 = configs["beta1"]
            self.beta2 = configs["beta2"]
        self.loss_history = []
        self.min_loss_model = None
        self.min_loss = (1 << 31) # int_max

    def train(self, l):
        # first clip the gradients from exploding gradient problem:
        for pair in l:
            np.clip(pair[1], -3, 3, out=pair[1])
        self.func(self, l)
        
    def step(self, i, loss, l):
        self.loss_history.append((i,loss))
        self.train(l)
        if self.min_loss >= loss:
            self.min_loss = loss
            self.min_loss_model = l
        
    
    def get_loss_history(self):
        return zip(*self.loss_history)
        