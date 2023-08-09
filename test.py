import numpy as np
from my_nn import *


## Test cost_function with regularization

def cost_with_regularization_test_case():

    np.random.seed(1)

    Y_asses = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1,1)
    
    weights = {"W1": W1, "W2": W2, "W3": W3}
    bias = {"b1": b1, "b2": b2, "b3": b3}

    A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    A1 = np.random.randn(2, 5)
    A2 = np.random.randn(3, 5)
    act_units = [A1, A2, A3]

    return act_units, Y_asses, weights, bias


## Test back prop with regularization

def backward_propagation_with_regularization_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])

    W1 = np.array([[-1.09989127, -0.17242821, -0.87785842],
         [ 0.04221375,  0.58281521, -1.10061918]])
    W2 = np.array([[ 0.50249434,  0.90085595],
         [-0.68372786, -0.12289023],
         [-0.93576943, -0.26788808]])
    W3 = np.array([[-0.6871727 , -0.84520564, -0.67124613]])
    b1 = np.array([[ 1.14472371],
         [ 0.90159072]])
     
    b2 = np.array([[ 0.53035547],
         [-0.69166075],
         [-0.39675353]])
    b3 = np.array([[-0.0126646]])

    weights = {"W1": W1, "W2": W2, "W3": W3}
    bias = {"b1": b1, "b2": b2, "b3": b3}

    A1 = np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],
         [ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]])
    A2 = np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])

    act_units = [A1, A2, A3]

    Z1 = np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
         [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]])
    Z2 = np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
         [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
         [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]])
    Z3 = np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]])

    cache = {"Z1": Z1, "Z2": Z2, "Z3": Z3}
    
    return X_assess, Y_assess, cache, act_units, weights, bias


def backward_propagation_with_regularization_test():
    
    X_assess, Y_assess, cache, act_units, weights, bias = backward_propagation_with_regularization_test_case()

    # create model
    model = my_MPL(num_inputs=X_assess.shape[0], hidden_layers= [2, 3], num_outputs= 1, activations= ['relu'],random_state=None)

    # set all parameters to the test values
    model.set_activation_units(act_units)
    model.set_parameters(weights, bias)

    # compute back prop
    model.backward_propagation_with_regularization(X_assess, Y_assess, lambd = 0.7)

    # expected outputs:
    expected_output = {'dZ3': np.array([[-0.59317598, -0.98370716,  0.16722898, -0.89881889,  0.40682402]]),
    'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
    'db3': np.array([[-0.38032981]]),
    'dA2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
            [ 0.50135568,  0.83143484, -0.14134288,  0.7596868 , -0.34384996],
            [ 0.39816708,  0.66030962, -0.11225181,  0.6033287 , -0.27307905]]),
    'dZ2': np.array([[ 0.40761434,  0.67597671, -0.11491519,  0.6176438 , -0.27955836],
            [ 0.        ,  0.        , -0.        ,  0.        , -0.        ],
            [ 0.        ,  0.        , -0.        ,  0.        , -0.        ]]),
    'dW2': np.array([[ 0.79276486,  0.85133918],
            [-0.0957219 , -0.01720463],
            [-0.13100772, -0.03750433]]),
    'db2': np.array([[0.26135226],
            [0.        ],
            [0.        ]]),
    'dA1': np.array([[ 0.2048239 ,  0.33967447, -0.05774423,  0.31036252, -0.14047649],
            [ 0.3672018 ,  0.60895764, -0.10352203,  0.5564081 , -0.25184181]]),
    'dZ1': np.array([[ 0.        ,  0.33967447, -0.05774423,  0.31036252, -0.        ],
            [ 0.        ,  0.60895764, -0.10352203,  0.5564081 , -0.        ]]),
    'dW1': np.array([[-0.25604646,  0.12298827, -0.28297129],
            [-0.17706303,  0.34536094, -0.4410571 ]]),
    'db1': np.array([[0.11845855],
            [0.21236874]])}
    print("gradients are : {} \n".format(model.grads))

    cond1 = np.allclose(model.grads["dW1"], expected_output["dW1"]) and np.allclose(model.grads["db1"], expected_output["db1"])
    cond2 = np.allclose(model.grads["dW2"], expected_output["dW2"]) and np.allclose(model.grads["db2"], expected_output["db2"])
    cond3 = np.allclose(model.grads["dW3"], expected_output["dW3"]) and np.allclose(model.grads["db3"], expected_output["db3"])


    if cond1 and cond2 and cond3:
        print("back prop with regularizatio tests passed")

def cost_function_with_regularization_test():
    
    act_units, Y_asses, weights, bias = cost_with_regularization_test_case()


    model = my_MPL(num_inputs=3, hidden_layers=[2, 3], num_outputs=1)
    model.set_parameters(weights, bias)
    model.set_activation_units(act_units=act_units)

    A3 = model.activation_units[-1]
    cost = model.cost_with_regularization(Y_asses, A3, lambd =0.1)
    print("cost with regularization is: {}".format(cost))

    if np.allclose(cost, np.float64(1.7864859451590758)) :
        print("cost with regularization test passed")


def gradient_check_test_case(seed):

    if seed:
        np.random.seed(seed)

    x = np.random.randn(4,3)
    y = np.array([[1, 1, 0]])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    W3 = np.random.randn(1,3) 
    b3 = np.random.randn(1,1) 
    weights = {"W1": W1, "W2": W2, "W3": W3}
    bias = { "b1": b1, "b2": b2, "b3": b3}
    
    return x, y, weights, bias

def gradient_check_test(x, y, params, epsilon = 1e-7):
    
    num_inputs = x.shape[0]
    num_outputs = y.shape[0]
    hidden = [params[0]["W"+str(i+1)].shape[0] for i in range(len(params[0].keys()) -1)]
    model = my_MPL(num_inputs= num_inputs, hidden_layers=hidden,
                   num_outputs=num_outputs)
    
    
    difference = model.gradient_check(x, y, params=params, epsilon=epsilon)
    expected_value = 1.1890913024229996e-07
    print("for epsilon {} the L2 difference in both gradients is: {}".format(epsilon, difference))
    # for seed = 1:
    #assert np.any(np.isclose(difference, expected_value)), "Wrong value. It is not one of the expected values"


# test minibatches:

def mini_batches_test(seed):

    np.random.seed(seed)
    mini_batch_size  = 64
    nx = 12288
    m = 148
    X =np.array([x for x in range(nx*m)]).reshape((m, nx)).T
    Y = np.random.randn(1, m) < 0.5

    mini_batches = random_mini_batches(X, Y, mini_batch_size=mini_batch_size, seed = 0)
    
    n_batches = len(mini_batches)

    assert n_batches == np.ceil(m / mini_batch_size), f"Wrong number of mini batches. {n_batches} != {np.ceil(m / mini_batch_size)}"
    for k in range(n_batches - 1):
        assert mini_batches[k][0].shape == (nx, mini_batch_size), f"Wrong shape in {k} mini batch for X"
        assert mini_batches[k][1].shape == (1, mini_batch_size), f"Wrong shape in {k} mini batch for Y"
        assert np.sum(np.sum(mini_batches[k][0] - mini_batches[k][0][0], axis=0)) == ((nx * (nx - 1) / 2 ) * mini_batch_size), "Wrong values. It happens if the order of X rows(features) changes"
    if ( m % mini_batch_size > 0):
        assert mini_batches[n_batches - 1][0].shape == (nx, m % mini_batch_size), f"Wrong shape in the last minibatch. {mini_batches[n_batches - 1][0].shape} != {(nx, m % mini_batch_size)}"

    assert np.allclose(mini_batches[0][0][0][0:3], [294912,  86016, 454656]), "Wrong values. Check the indexes used to form the mini batches"
    assert np.allclose(mini_batches[-1][0][-1][0:3], [1425407, 1769471, 897023]), "Wrong values. Check the indexes used to form the mini batches"

    print("\033[92mAll tests passed!")





if __name__ == "__main__":

    #test cost with regularization term
    cost_function_with_regularization_test()

    # test backprop with regularization
    backward_propagation_with_regularization_test()

    # test gradient computation
    x, y, weights, bias = gradient_check_test_case(None)
    params = [weights, bias]
    gradient_check_test(x, y, params)

    # test minibatches

    mini_batches_test(seed =1)





    