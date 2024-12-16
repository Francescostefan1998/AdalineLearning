import numpy as np

class AdalineGD:
   
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]) # all the weights, it is a list "kind of a random list"
        self.b_ = np.float64(0.)
        self.losses_ = []
        for i in range(self.n_iter):
                 net_input = self.net_input(X)
                 output = self.activation(net_input)
                 errors =(y- output) # label - firs net input calculate and so on for each label with the subseqent net input calculated
                 self.w_ += self.calculateUpdateWeights(self.eta, errors, X)  # 
                 self.b_ += self.eta * 2.0 * errors.mean()
                 loss = (errors**2).mean() # the .mean() calculate the average (all valuse/n of values)
                 self.losses_.append(loss)
        return self
    
    def net_input(self, X):
             return np.dot(X, self.w_) + self.b_ 
    def activation(self, X):
             return X
        
    def predict(self, X):
             return np.where(self.activation(self.net_input(X)) >= 0.05, 1, 0)
 
    def calculateUpdateWeights(self, eta, errors, X):
        n_features = X.shape[1]  
        n_samples = X.shape[0]  
        weight_updates = [0.0] * n_features  # Start with zeros for all weights
        for feature_idx in range(n_features):  # For each feature (column in X)
            feature_error_sum = 0.0  # Initialize accumulator for this feature
            for sample_idx in range(n_samples):  # For each sample (row in X)
                feature_value = X[sample_idx][feature_idx]  # Value of this feature for the sample
                error = errors[sample_idx]  # Corresponding error for the sample
                feature_error_sum += feature_value * error  # Add to feature's contribution
            
            feature_error_mean = feature_error_sum / n_samples  # Mean of feature's error contributions
            weight_updates[feature_idx] = eta * 2.0 * feature_error_mean  # Apply scaling factors
        

        return weight_updates
