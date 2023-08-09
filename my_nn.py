import numpy as np
import copy
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


# utils functions

def parameters(weights, bias):

    """
    Takes two dicts and join them inyo a big dict. Returns the dict and the keys
    
    """
    keys = [[keyw, keyb] for (keyw, keyb) in zip(weights.keys(), bias.keys())]
    keys = sum(keys, [])

    params = {**weights, **bias}
    return params, keys


def dictionary_to_vector(params, keys=None):

    """
    this fucntion unrolled the matrices in the dict params to a big vector

    Inputs:

    params: dict. A dictionary with the matrices W's and b's
    keys = list of keys or None. If None, then keys is set as params.keys()

    Outputs:
    theta: ndarray of shape (N, 1) with the W's and b's unrolled (W1,b1,W2,b2....)
    """

    if not keys:
        keys = params.keys()


    count = 0
    for key in keys:
        new_vector = np.reshape(params[key], -1)

        if count == 0:
            theta = new_vector
        else:
            theta = np.hstack((theta, new_vector))
        
        count += 1

    return theta

def vector_to_dictionary(theta, dimensions):

    indices = [int(np.prod(item) + item[0]) for item in dimensions]    
    
    Ws = {}
    Bs = {}
    s = 0
    for j in range(len(indices)):
        l = indices[j]
        dimW = np.prod(dimensions[j])
        dimb = dimensions[j][0]
        print(s, s +l )
        theta_aux = theta[s: s+l]
        print("from {} to {} are for W".format(theta_aux[0], theta_aux[-1]))
        w = theta_aux[: dimW].reshape(dimensions[j])
        b = theta_aux[dimW : ].reshape(-1,1)
        print("W = {}".format(w))
        print("b = {}".format(b))
        s = s +l

        Ws.update([("W"+str(j+1), w)])
        Bs.update([("b"+str(j+1), b)])

    return Ws, Bs

## utils for optimization methods:

def random_mini_batches(X, Y, mini_batch_size, seed):

    """
    Creates a list of random minibatches from data (X,Y)

    Arguments:
    X: ndarray. Input data for trainig of shape (num_inputs, num_examples)
    Y: ndarray. Output data for training  of shape (num_outputs, num_examples)
    mini_batch_size: int. Tipically a power of 2. The size of the minibatches
    seed: int. 

    Returns:
    mini_batches: list. List with the minibatches of size mini_batch_size
    """
    np.random.seed(seed)
    m = X.shape[1]   # num of examples in the data
    permutations = list(np.random.permutation(m))   #gives a permutation of set {1,2,..,m}

    X_shuffled = X[:, permutations]    # inputs permuted
    Y_shuffled = Y[:, permutations]    # outpus permuted accordingly

    # create the minibatches
    mini_batches = []

    num_mini_batches = int(np.floor(m/mini_batch_size))

    for k in range(num_mini_batches):
        
        mini_batch_X = X_shuffled[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = Y_shuffled[:, k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batches.append((mini_batch_X, mini_batch_Y))

    # handle the case of m %% mini_batch_size != 0
    if m % mini_batch_size != 0 :
        mini_batch_X = X_shuffled[:, (k+1)* mini_batch_size : ]
        mini_batch_Y = Y_shuffled[:, (k+1)* mini_batch_size : ]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches



def initialize_velocity(weights, bias, num_layers):

    """
    initialize first momentum params v
    """
    v = {}   # velocity (moving avg of gradients) dicttionary
    

    for l in range(num_layers):
        v["dW" + str(l+1)] = np.zeros((weights["W" +str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((bias["b" +str(l+1)].shape))
    
    return v

def initialize_Adam(weights, bias, num_layers):

    """
    initialize second momentum params s
    """
    s = {}   # velocity (moving avg of gradients) dicttionary
    

    for l in range(num_layers):
        s["dW" + str(l+1)] = np.zeros((weights["W" +str(l+1)].shape))
        s["db" + str(l+1)] = np.zeros((bias["b" +str(l+1)].shape))
    
    return s




## activation functions and its derivatives:

def identity(X):

    return X

def sigmoid(X):

    return 1/(1 + np.exp(-X))

def tanh(X):
    return np.tanh(X)

def softmax(X):

    expX = np.exp(X) 
    return expX / np.sum(expX, axis = 1, keepdims = True)

def relu(X):

    return np.maximum(X, 0)

def der_identity(X):

    return None

def der_sigmoid(X):

    return sigmoid(X)*(1 - sigmoid(X))

def der_relu(X):

    return 1* (X > 0)

def der_tanh(X):

    return 1 - np.power(tanh(X), 2)#(tanh(X))**2





ACTIVATIONS = {
    "identity": identity,
    "sigmoid" : sigmoid,
    "relu" : relu,
    "tanh" : tanh,
    "softmax": softmax
}

DERIVATIVE_ACTIVATIONS = {
    "identity": der_identity,
    "sigmoid" : der_sigmoid,
    "relu" : der_relu,
    "tanh" : der_tanh,
}

class my_MPL():

    def __init__(self, num_inputs,
                  hidden_layers = [4], num_outputs = 1, activations = ['relu'],cost = 'Hinge Loss', random_state = 3):
        """
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
            activation (string): activation function in 
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.activations = activations
        self.cost = cost

        # representation of layers: a list with the numbers of activaton uints in each layer
        layers = [num_inputs] + hidden_layers + [num_outputs]   
        self.layers = layers

        num_layers = len(layers) - 1  # the number of matrices W (or bias vectors b)
        self.num_layers = num_layers

        # Representation the matrices W and biases b: for the purpose of understanding use a dict
        weights = {}   
        bias = {}
        if random_state :
            np.random.seed(random_state)

        for l in range(num_layers):
            w = np.sqrt(2/layers[l]) * np.random.randn(layers[l+1], layers[l])
            b = np.zeros((layers[l+1], 1))
            weights["W" + str(l+1)] = w
            bias["b" + str(l+1)] = b

        self.weights = weights
        self.bias = bias


        # save activation units per layer
        activation_units = []
        for l in range(num_layers + 1):
            a = np.zeros(layers[l]) 
            activation_units.append(a)

        self.activation_units = activation_units

        # save derivatives per layer
        grads = {}
        for l in range(self.num_layers):
            dW = np.zeros((layers[l+1], layers[l]))
            db = np.zeros((layers[l + 1], 1))
            grads["dW" + str(l+1)] = dW
            grads["db" + str(l+1)] = db
        
        self.grads = grads

        
    def initialization(self, method = "he", random_state = None):
        
        weights = {}   
        bias = {}
        num_layers = self.num_layers
        layers = self.layers

        if not random_state:
            np.random.seed(random_state)

        if method == 'he':
            for l in range(num_layers):
                w = np.random.randn(layers[l+1], layers[l]) * np.sqrt(2/layers[l])
                b = np.zeros((layers[l+1], 1))
                weights["W" + str(l+1)] = w
                bias["b" + str(l+1)] = b
        
        if method == 'random':

            for l in range(num_layers):
                w = np.random.randn(layers[l+1], layers[l]) * 0.01
                b = np.zeros((layers[l+1], 1))
                weights["W" + str(l+1)] = w
                bias["b" + str(l+1)] = b


        self.weights = weights
        self.bias = bias



    def get_parameters(self):

        """
        this method allows user to get the weights and the bias
        """

        for l in range(len(self.weights)):
            print("the weight matrix W{} is :\n {}".format(l+1, self.weights["W" + str(l+1)]))
            print("the bias b{} is :\n {} ".format(l+1, self.bias["b" + str(l+1)]))

        params = [self.weights, self.bias]

        return params
    
    def set_parameters(self, weights, bias):

        """
        this method allows the user to set weight and bias matrices as the matrices given by weights and bias

        inputs
        -------

        weights: dict. Dictionary with the weight matrices W1,..,W[L+1]
        bias: dict. Dictionary with the bias matrices b1,..., b[L+1]
        """

        self.weights = weights
        self.bias = bias

    def set_activation_units(self, act_units):

        """
        this method sets a desired activation units. Used to check the performace of the class my_MPL
        """

        self.activation_units = act_units
        
        

    def forward_propagation(self, X):
        
        """
        Argument:
        X -- input data of size (n_x, m)
        
        Returns:
        act_units: list. The activation units per each layer
        """
        # Retrieve each parameter from the dictionary "parameters"
      
        
        # Implement Forward Propagation :

        act_units = [X]

        cache = {}  # a dict with the linear activations Z

        for l in range(1, self.num_layers):
            Z = np.dot(self.weights["W" + str(l)], act_units[l-1]) +  self.bias["b" + str(l)]
            cache["Z" + str(l)] = Z 
            A = ACTIVATIONS[self.activations[0]](Z)
            act_units.append(A)
        
        A = act_units[-1]
        Z = np.dot(self.weights["W" + str(self.num_layers)], A) +  self.bias["b" + str(self.num_layers)]
        cache["Z" +str(self.num_layers)] = Z
        A = ACTIVATIONS['sigmoid'](Z)

        act_units.append(A)
        
        self.activation_units = act_units

        
        return cache, act_units

    def backward_propagation(self, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        self: to obtain weights and bias
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """

        
        m = X.shape[1]
        cache, act_units = self.forward_propagation(X)
        L = self.num_layers -1
        A_L_plus1 = act_units[-1]
        Z_L_plus1 = cache["Z" + str(L+1)]
        AL = act_units[-2]

        # dA[L+1] = der of cost function w.r.t A[L+1]

        dAL_plus1 =  -(np.divide(Y, A_L_plus1) - np.divide(1 - Y, 1 - A_L_plus1))
        dZL_plus1 = dAL_plus1 * der_sigmoid(Z_L_plus1)
        dWL_plus1 = np.dot(dZL_plus1, AL.T) / m
        dbL_plus1 = np.sum(dZL_plus1, axis = 1, keepdims = True) / m

        der_cache = {}
        der_cache["dZ" + str(L+1)] = dZL_plus1

        #grads
        self.grads["dW" + str(L + 1)] = dWL_plus1
        self.grads["db" + str(L + 1)] = dbL_plus1
        

        for l in reversed(range(1, L+1)):

            # dA[l] = W[l+1].T @ dZ[l+1]
            # dZ[l] = dA[l]* g'(Z[l])
            # dW[l] = dZ[l] @ A[l-1].T / m
            # db[l] =  sum(dZ[l], axis = 1, keepdims =True)/m


            W = self.weights["W" + str(l+1)]
            A_prev = act_units[l-1]
            dA = np.dot(W.T, der_cache["dZ" + str(l+1)])
            dZ = dA * DERIVATIVE_ACTIVATIONS[self.activations[0]](cache["Z" + str(l)])
            dW = np.dot(dZ, A_prev.T) /m
            db = np.sum(dZ, axis = 1, keepdims = True) / m

            self.grads["dW" + str(l)] = dW
            self.grads["db" + str(l)] = db
            der_cache["dZ" + str(l)] = dZ
       
    
        return der_cache
    

    def backward_propagation_with_regularization(self, X, Y, lambd):

        """
        this method implements back prop algorithm for a cost function with regularization term
        
        Arguments:
        self: to obtain weights and bias
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """

        
        m = X.shape[1]
        cache, act_units = self.forward_propagation(X)
        L = self.num_layers -1
        A_L_plus1 = act_units[-1]
        Z_L_plus1 = cache["Z" + str(L+1)]
        AL = act_units[-2]
        W_Lplus1 = self.weights["W" + str(L+1)]
        WL = self.weights["W" + str(L)]


        # dA[L+1] = der of cost function w.r.t A[L+1]

        dAL_plus1 =  -(np.divide(Y, A_L_plus1) - np.divide(1 - Y, 1 - A_L_plus1))
        dZL_plus1 = dAL_plus1 * der_sigmoid(Z_L_plus1)
        dWL_plus1 = np.dot(dZL_plus1, AL.T) / m + lambd /m * W_Lplus1
        dbL_plus1 = np.sum(dZL_plus1, axis = 1, keepdims = True) / m

        der_cache = {}
        der_cache["dZ" + str(L+1)] = dZL_plus1

        #grads
        self.grads["dW" + str(L + 1)] = dWL_plus1
        self.grads["db" + str(L + 1)] = dbL_plus1

        

        for l in reversed(range(1, L+1)):

            # dA[l] = W[l+1].T @ dZ[l+1]
            # dZ[l] = dA[l]* g'(Z[l])
            # dW[l] = dZ[l] @ A[l-1].T / m
            # db[l] =  sum(dZ[l], axis = 1, keepdims =True)/m


            Wl_plus1 = self.weights["W" + str(l+1)]
            Wl = self.weights["W" + str(l)]
            A_prev = act_units[l-1]
            dA = np.dot(Wl_plus1.T, der_cache["dZ" + str(l+1)])
            dZ = dA * DERIVATIVE_ACTIVATIONS[self.activations[0]](cache["Z" + str(l)])
            dW = np.dot(dZ, A_prev.T) /m + lambd /m * Wl
            db = np.sum(dZ, axis = 1, keepdims = True) / m

            self.grads["dW" + str(l)] = dW
            self.grads["db" + str(l)] = db
            der_cache["dZ" + str(l)] = dZ
       
    
        return der_cache


# Optimization methods: Gradient descent, stochastic and minibatch gradient descent
    
    def gradient_descent(self, learning_rate):

        """
        updates parameteres W and b in place 
        """

        for l in range(self.num_layers):

            weights = self.weights
            bias =  self.bias

            dW = self.grads["dW" + str(l+1)]
            db = self.grads["db" + str(l+1)]
            weights["W" + str(l+1)] = weights["W" + str(l+1)]  - learning_rate * dW
            bias["b" + str(l+1)] =  bias["b" + str(l+1)] - learning_rate * db




    def update_with_momentum(self, learning_rate, beta):
        """
        updates the parameters (in place) using momentum method:
        v["dW"] = beta* v["dW"] + (1-beta)*dW
        W = W - learning_rate*v["dW"]

        v["db"] = beta* v["db"] + (1-beta)*db
        b = b - learning_rate*v["db"]

        Arguments: 
        learninig_rate: float. the size of the step to tek in the optimization
        beta: float in (0,1). The weight for the moving average to compute momentum
        """

        # initialize velocity params v:
        weights = self.weights
        bias = self.bias
        num_layers =  self.num_layers

        v = {}   # velocity (momentum) dicttionary

        for l in range(num_layers):
            v["dW" + str(l+1)] = np.zeros((weights["W" +str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((bias["b" +str(l+1)].shape))
     
        # compute momentum and update
        for l in range(num_layers):

            v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1- beta)*self.grads["dW" + str(l+1)] 
            v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1- beta)*self.grads["db" + str(l+1)]

            weights["W" + str(l+1)] = weights["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
            bias["b" + str(l+1)] = bias["b" + str(l+1)]  - learning_rate * v["db" + str(l+1)]

    def Adam(self, learning_rate, t,beta1=0.9, beta2=0.999, epsilon=1e-8):

        """
        updates the parameters (in place) using momentum method:
        v["dW"] = beta* v["dW"] + (1-beta)*dW
        W = W - learning_rate*v["dW"]

        v["db"] = beta* v["db"] + (1-beta)*db
        b = b - learning_rate*v["db"]

        Arguments: 
        learninig_rate: float. the size of the step to tek in the optimization
        beta: float in (0,1). The weight for the moving average to compute momentum
        """

        weights = self.weights
        bias = self.bias
        num_layers =  self.num_layers

        # initialize first and second momentum params v and s:

        v = initialize_velocity(weights, bias, num_layers)   # velocity (moving avg of gradients) dicttionary
        s = initialize_Adam(weights, bias, num_layers)   # movin avg of squared gradients dictionary

        
        # compute momentum and update
        v_corrected = {}
        s_corrected = {}

        for l in range(num_layers):
            # first momentum
            v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1- beta1)*self.grads["dW" + str(l+1)] 
            v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1- beta1)*self.grads["db" + str(l+1)]

            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] /(1 - beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)] /(1 - beta1**t)

            # second momentum
            s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1- beta2)*self.grads["dW" + str(l+1)]**2
            s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1- beta2)*self.grads["db" + str(l+1)]**2

            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] /(1 - beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] /(1 - beta2**t)

            weights["W" + str(l+1)] = weights["W" + str(l+1)] -learning_rate * v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
            bias["b" + str(l+1)] = bias["b" + str(l+1)] -learning_rate * v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]))


        

    def cost_function(self, Y, output):

        m =  Y.shape[1]
        if self.cost == 'Hinge Loss':
            cost = -(np.dot(Y, np.log(output).T) + np.dot(1 - Y, np.log(1 - output).T)) / m
        
        elif self.cost == 'MSE':
            cost = np.average(Y - output)

        else:
            raise("Undefined cost function. Make sure you've chosen either Hinge Loss or MSE")

        cost = float(np.squeeze(cost))
        return cost
    

    def cost_with_regularization(self, Y, output, lambd):
        """
        this method computes cost with regularization term
        """

        if lambd == 0:
            cost = self.cost_function(Y, output)
        
        else:
            m = Y.shape[1]
            L2_regularization = lambd/(2*m) * sum([np.sum(self.weights[key]**2) for key in self.weights.keys()])
            cost = self.cost_function(Y, output) + L2_regularization

        return cost





    # def train(self, X, Y, num_iter, learning_rate, lambd = 0, keep_prob = 1, num_mini_batches = 1, optimizer = "GD"):

        # costs = []  # to keep track of the cost function values

        # for i in range(num_iter):

        #     # forward propagate
        #     self.forward_propagation(X)
        #     AL_plus1 = self.activation_units[-1]
        #     #print(AL_plus1)


        #     # compute cost 
            
        #     cost = self.cost_with_regularization(Y, AL_plus1, lambd)
            
        #     costs.append(cost)   # keep track of cost
             
        #     # backward propagate
        #     if lambd == 0:
        #         self.backward_propagation(X, Y)

        #     else: 
        #         self.backward_propagation_with_regularization(X, Y, lambd)

        #     # update parameters 
        #     self.gradient_descent(learning_rate)
            
        #     if i % 1000 ==0 or i == num_iter:
        #         print("Cost function : {} at iter {}".format(cost, i))

        # print("Training complete!")
        # print("============")

        # return costs
    

    def train(self, X, Y, num_epochs, learning_rate, mini_batch_size, lambd = 0,   optimizer = "GD"):


        # initialize parameters for Adam and momentum:
        if optimizer == "GD":
            pass  # no initalization required
        
        elif optimizer == "Adam":
            v = initialize_velocity(self.weights, self.bias, self.num_layers)
            s = initialize_Adam(self.weights, self.bias, self.num_layers)
        
        elif optimizer == "Momentum":
            v = initialize_velocity(self.weights, self.bias, self.num_layers)


        costs = []  # to keep track of the cost function values
        m = X.shape[1]  # num_trainig
        seed = 10
        t = 0

        # optimization loop
        for i in range(num_epochs):


            # define random mini batches
            seed = seed + 1
            mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=seed)
            total_cost = 0
            
            # loop in minibatches

            for mini_batch in mini_batches:

                (mini_batch_X, mini_batch_Y) = mini_batch

                # forward propagate
                self.forward_propagation(mini_batch_X)
                AL_plus1 = self.activation_units[-1]
                # compute cost 
                cost = self.cost_with_regularization(mini_batch_Y, AL_plus1, lambd)
                total_cost += cost
            
             
                # backward propagate
                if lambd == 0:
                    self.backward_propagation(mini_batch_X, mini_batch_Y)

                else: 
                    self.backward_propagation_with_regularization(mini_batch_X, mini_batch_Y, lambd)

                # update parameters 

                if optimizer == "GD":

                    self.gradient_descent(learning_rate)

                elif optimizer == "Adam":
                    
                    t = t + 1   # Adam counter
                    self.Adam(learning_rate, t)
                
                elif optimizer == "Momentum" :
                    pass

            avg_cost = total_cost /m
            
            # print cost every 1000 epchs
            if i % 1000 == 0:
                print("Cost after epoch {} is {}".format(i, avg_cost))
            
            if i % 100 == 0:
                costs.append(avg_cost)


        return costs




    def predict(self, X):

        """
        This method is used to predict the results of a  L-layer neural network.
        
        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model
        
        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        predictions = np.zeros((1, m))

        # forward prop
        cache, act_units = self.forward_propagation(X)
        #act_units = self.activation_units

        probas = act_units[-1]
        predictions = (1 - predictions) * (probas > 0.5)

        return predictions
    
    def accuracy(self, X, Y):

        predictions = self.predict(X)    
        print("Accuracy on this set is: {}".format(np.average(predictions == Y)))



# tests

# Gradient checking:

    def gradient_check(self, X, Y, params, epsilon, verbose = False):

        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
        
        Arguments:
        parameters -- python list containing 2 dictionaries with your parameters Weights and Bias
        grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters 
        X -- input datapoint, of shape (input size, number of examples)
        Y -- true "label"
        epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
        
        Returns:
        difference -- difference (2) between the approximated gradient and the backward propagation gradient
        """

        # set the parameters to the model equal to the given ones in the list parameters

        weights = params[0]
        bias = params[1]
        self.set_parameters(weights, bias)
        dimensions = [(weights[key].shape[0], weights[key].shape[1]) for key in weights.keys()]
        print(self.weights)
        # to compute the gradient at these parameters lets forward -> backward
        self.forward_propagation(X)
        self.backward_propagation(X, Y)

        gradients = self.grads

        # convert gradient to a long vector
        gradient_vector = dictionary_to_vector(gradients) 

        ## Compute the approximate gradient 
        
        
        # get one big dict out of weights and bias
        big_params, keys = parameters(weights, bias)
        parameters_values = dictionary_to_vector(big_params, keys)  # big theta

        dimension = len(parameters_values)
        
        # initialize J+ and J- and grad_approx
        J_plus = np.zeros(dimension)
        J_minus = np.zeros(dimension)
        grad_approx = np.zeros(dimension)

        for i in range(dimension):
            # compute cost J+ at theta+ 
            theta_plus = np.copy(parameters_values)
            theta_plus[i] = theta_plus[i] + epsilon

            weights_plus, bias_plus = vector_to_dictionary(theta_plus, dimensions)
            self.set_parameters(weights_plus, bias_plus)
            _, act_units_plus = self.forward_propagation(X)
            output_plus = act_units_plus[-1]
            J_plus[i] = self.cost_function(Y, output_plus)

            # compute cost J- at theta-
            theta_minus = np.copy(parameters_values)
            theta_minus[i] = theta_minus[i] - epsilon

            weights_minus, bias_minus = vector_to_dictionary(theta_minus, dimensions)

            self.set_parameters(weights_minus, bias_minus)
            _, act_units_minus = self.forward_propagation(X)
            output_minus = act_units_minus[-1]
            J_minus[i] = self.cost_function(Y, output_minus)

            # approx grads
            grad_approx[i] = (J_plus[i] - J_minus[i])/(2*epsilon)

        numerator = np.linalg.norm(gradient_vector - grad_approx)
        denominator = np.linalg.norm(gradient_vector) + np.linalg.norm(grad_approx)

        difference = numerator / denominator

        return difference





# Examples:


## create some datasets

class flower_dataset():

    def __init__(self, num_samples = 400, num_classes = 2, radius = 4, num_petals =  4, random_state = 1):

        self.num_samples = num_samples
        self.num_classes = num_classes
        self.radius = radius
        self.num_petals = num_petals



        np.random.seed(random_state)

        D = 2
        N = int(num_samples /2)  # num of points per class
        X = np.zeros((num_samples, D))
        Y = np.zeros((num_samples, 1), dtype='uint8')

        for j in range(D):
            ix = range(N*j, N*(j+1))
            theta = np.linspace(j*np.pi, (j+1)*np.pi/2, N) + 0.2*np.random.randn(N)
            r = radius *np.sin(num_petals * theta) + np.random.randn(N)*0.2
            X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
            Y[ix] = j


        X = X.T
        Y = Y.T
        self.points = X
        self.labels = Y

    def plot_flower(self):

        plt.figure(figsize = (16,9))
        plt.scatter(self.points[0,:], self.points[1, :], c = self.labels, s =40,
                     cmap = plt.cm.Spectral)



# if __name__ == "__main__" :

#     # create dataset

#     dataset = flower_dataset()

#     # plot dataset
#     dataset.plot_flower()


#     X = dataset.points
#     Y = dataset.labels

#     print("num of inputs is {}, and samples is {}".format(X.shape[0],X.shape[1]))


      
#     model = my_MPL(num_inputs=X.shape[0], hidden_layers=[4], num_outputs=Y.shape[0], activations=['tanh'], cost ='Hinge Loss' ,random_state=3)
#     model.train(X, Y, num_iter=10000, learning_rate=1.2)

#     model.accuracy(X, Y)
#     model.get_parameters()


            
