import numpy as np
import copy
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt


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
                  hidden_layers = [4], num_outputs = 2, activations = ['relu'],cost = 'Hinge Loss', random_state = 3):
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
        num_layers = len(layers) - 1  # the number of matrices W (or bias vectors b)
        self.num_layers = num_layers

        # Representation the matrices W and biases b: for the purpose of understanding use a dict
        weights = {}   
        bias = {}
        np.random.seed(random_state)
        for l in range(num_layers):
            w = 0.01 * np.random.randn(layers[l+1], layers[l])
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

        



    def get_parameters(self):

        for l in range(len(self.weights)):
            print("the weight matrix W{} is :\n {}".format(l+1, self.weights["W" + str(l+1)]))
            print("the bias b{} is :\n {} ".format(l+1, self.bias["b" + str(l+1)]))

        params = [self.weights, self.bias]

        return params
    
    def set_parameters(self, weights, bias):

        """
        sets as weight and bias matrices the matrices given by weights and bias

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

        # dZ[L] is computed via the derivative of sigmoid:
        W = self.weights["W" + str(L+1)]
        dAL = np.dot(W.T, dZL_plus1)
        dZL = dAL * der_sigmoid(cache["Z" +str(L)])
        dWL = np.dot(dZL, act_units[-3].T) / m
        dbL = np.sum(dZL, axis =1, keepdims = True) / m

        self.grads["dW" + str(L)] = dWL
        self.grads["db" + str(L)] = dbL
        der_cache["dZ" + str(L)] = dZL

        

        for l in reversed(range(1, L)):

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

    def train(self, X, Y, num_iter, learning_rate):

        costs = []  # to keep track of the cost function values

        for i in range(num_iter):

            # forward propagate
            self.forward_propagation(X)
            AL_plus1 = self.activation_units[-1]
            #print(AL_plus1)


            # compute cost 
            cost = self.cost_function(Y, AL_plus1)
            
            # backward propagate
            self.backward_propagation(X, Y)

            # update parameters 
            self.gradient_descent(learning_rate)
            
            if i % 1000 ==0 or i == num_iter:
                print("Cost function : {} at iter {}".format(cost, i))

        print("Training complete!")
        print("============")



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



# create some datasets

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





#if __name__ == "__main__" :

    # create dataset

    # dataset = flower_dataset()

    # # plot dataset
    # dataset.plot_flower()


    # X = dataset.points
    # Y = dataset.labels

    # print("num of inputs is {}, and samples is {}".format(X.shape[0],X.shape[1]))


      
    # model = my_MPL(num_inputs=X.shape[0], hidden_layers=[4], num_outputs=Y.shape[0], activations=['tanh'], random_state=3)

    # model.train(X, Y, num_iter=10000, learning_rate=1.2)

    # model.accuracy(X, Y)
    # model.get_parameters()


            
