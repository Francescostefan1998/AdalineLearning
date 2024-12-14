import numpy as np

class AdalineGD:
    """ADAptive LInear NEuron classifier.
    
    Parameters
    ------------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset
    random_state: int
        Random number generator seed for random weight initialization
    
    Attributes
    ----------
    w_: 1d-array
        Weights after fitting.
    b_: Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function valuse in each epoch.
    
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        print(f'Initializing : with eta = {eta}, n_iter = {n_iter}, random_state ={random_state}')

    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and n_features is the number of features
        y : {array-like}, shape = [n_examples] Target values.

        Return 
        --------------------------------
        self: object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # all the weights, it is a list "kind of a random list"
        self.b_ = np.float64(0.)
        self.losses_ = []
        print('FIT METHOD')
        print(f'weights = {self.w_}')
        print(f'bias =  {self.b_}')
        print(f'self losses = {self.losses_}')
        for i in range(self.n_iter):
                 net_input = self.net_input(X)
                 print(f'net_input = {net_input}')
                 output = self.activation(net_input)
                 print(f'output = {output}')
                 errors =(y- output)
                 print(f'errors = {errors}')
                 print(f'weights before = {self.w_}')
                 self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]  # 
                 print(f'weights after = {self.w_}')
                 print(f'weights before = {self.b_}')
                 self.b_ += self.eta * 2.0 * errors.mean()
                 print(f'weights before = {self.b_}')
                 loss = (errors**2).mean() # the .mean() calculate the average (all valuse/n of values)
                 print(f'loss = {loss}')
                 self.losses_.append(loss)
                 print(f'losses = {self.losses_}')
        return self
    
    def net_input(self, X):
             """Calculate net input"""
             return np.dot(X, self.w_) + self.b_
        
    def activation(self, X):
             """Compute linear activation"""
             return X
        
    def predict(self, X):
             """Returns class label after unit step"""
             return np.where(self.activation(self.net_input(X)) >= 0.05, 1, 0)
    
    # note that this line we are using is the implementation of the following code 
    # self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]  # 
    # for w_j in range(self.w_.shape[0]):
    #   self.w_[w_j] += self.eta * (2.0 * (X[:, w_j]*errors)).mean()


    