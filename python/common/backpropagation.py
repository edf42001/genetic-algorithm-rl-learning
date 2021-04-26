import numpy as np


def create_model(layers):
    """Creates network matrices and randomly initializes starting model parameters"""

    model = []  # Empty current population

    # Add a layer for every layer in the network
    for i in range(1, len(layers)):
        shape = (layers[i], layers[i-1])
        u = 0
        std = 1 / np.sqrt(layers[i-1])  # "Xavier" initialization

        # For this layer, create a random weight matrix with the given u and std
        model.append(np.random.normal(u, std, shape))

    return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def entropy_of_probabilities(x):
    return -np.sum(x * np.log(x))


def policy_feed_forward(model, input):
    hidden = model[0].dot(input)
    result = model[1].dot(hidden)

    return result, hidden


def softmax_jacobian(x):
    """
    The matrix has x(1-x) on the diagonal, and -xixj on the off diagonals
    :param x: The output values of the softmax. The jacobian can be written in terms of these.
                This is common in exponential functions
    :return: nxn jacobian matrix
    """
    n = len(x)
    jac = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jac[i, j] = x[i] * (1 - x[i])
            else:
                jac[i, j] = -x[i] * x[j]

    return jac


def weight_matrix_jacobian(x, weight_shape):
    """
    :param x: the inputs that were fed to the weight matrix, size m
    :param weight_shape: weight matrix, size m x n
    :return: Jacobian matrix, size n x (m*n)
    """
    n = weight_shape[0]
    m = weight_shape[1]
    assert m == len(x), "Dimensions must match"

    jac = np.zeros(shape=(n, m*n))

    for i in range(n):
        jac[i, i*m:(i+1)*m] = x

    return jac


def cross_entropy_jacobian(y, label):
    """
    :param y: The output of the softmax of this layer, the probability logits
    :return: Cross entropy jacobian: 0 everywhere, -1/yi for the value of true label
    """

    jac = np.zeros_like(y, dtype='float64')
    jac[label == 1] = -1.0 / y[label == 1]

    return jac


def complex_softmax_layer_jacobian(x, y, label, weight_shape):
    """
    Combines the three operations to get the jacobian of the loss function with respect to the weights matrix
    :param x:
    :param y:
    :param label:
    :param weight_shape:
    :return:
    """
    ce_jac = cross_entropy_jacobian(y, label)
    softmax_jac = softmax_jacobian(y)
    weights_jac = weight_matrix_jacobian(x, weight_shape)
    return ce_jac.dot(softmax_jac.dot(weights_jac)).reshape(len(x), len(y))


def softmax_layer_jacobian(x, loss):
    """
    :param x: The inputs to this layer (batch_size x n)
    :param loss: The actual action taken minus the softmax outputs from this layer (batch_size x m)
    Calculates the jacobian operator with respect to weights on the output cross entropy loss, but with a simpler method
    :return:
    """

    return np.dot(loss.T, x)


def softmax_layer_jacobian_wrt_inputs(weights, loss):
    # Equivalent to:
    # softmax_jac = softmax_jacobian(y)
    # cross_entropy_jac = cross_entropy_jacobian(y, label)
    # return cross_entropy_jac.dot(softmax_jac.dot(weights))

    # (batch x m), (m X h) -> (batch_size x h)
    return np.dot(loss, weights)


def previous_layers_weights_jacobian(x, weights, loss):
    """
    :param x: (batch size x n)
    :param weights: (m x h)
    :param loss: (batch_size x m)
    :return: previous layers weights (h x n)
    """

    # d_hidden: gradient of hidden layer, (batch_size x h)
    d_hidden = softmax_layer_jacobian_wrt_inputs(weights, loss)

    return np.dot(d_hidden.T, x)


def entropy_loss_jacobian(y):
    # Entropy = -sum(pi*log(pi))
    # Derivative of entropy with respect to pi = -1 - log(pi)
    jac = - (1 + np.log(y))
    return jac


def weights_gradient_wrt_entropy_loss(x, y):
    """
    Finds the gradients of the weight in the layer with respect to softmax output entropy
    :param x: Stacked inputs to this layer (N x n)
    :param y: Stacked softmax outputs of this layer (N x m)
    :return: gradients 1 x (mxn), but reshaped to the weights shape, (m x n)
    """

    n = x.shape[1]
    m = y.shape[1]

    grad = np.zeros((1, m*n))

    # Just do this in a loop, I can't think of a more efficient vectorized way yet
    for xi, yi in zip(x, y):
        entropy_jac = entropy_loss_jacobian(yi)  # (1 x m)
        softmax_jac = softmax_jacobian(yi)  # (m x m)
        weights_jac = weight_matrix_jacobian(xi, weight_shape=(m, n))  # (m x (mxn))
        entropy_softmax_jac = entropy_jac.dot(softmax_jac)  # (1 x m)
        entropy_softmax_weights_jac = entropy_softmax_jac.dot(weights_jac)  # (1 x (mxn))

        grad += entropy_softmax_weights_jac

    return grad.reshape((m, n))


def previous_layer_jacobian(weights):
    """
    Finds the jacobian of the input values with respect to the output values
    :param y: The input values (1xn)
    :param weights: The weights of this layer (m x n)
    :return: Jacobian (n x m)
    """

    # The jacobian is literally just the weights matrix
    return weights


def first_layer_weights_gradient_wrt_entropy_loss(x, h, y, weights):
    """
    The gradients of the weight in the first network layer with respect to softmax output entropy
    :param x: Inputs to the network (1 x n)
    :param h: Hidden layer values (1 x q)
    :param y: Softmax outputs of the network (1 x m)
    :param weights: Second layer weights (m x q)
    :return: gradients 1 x (qxn), but reshaped to the weights shape, (q x n)
    """

    q = h.shape[1]
    n = x.shape[1]

    grad = np.zeros((1, q*n))

    # Just do this in a loop, I can't think of a more efficient vectorized way yet
    for xi, yi in zip(x, y):
        entropy_jac = entropy_loss_jacobian(yi)  # (1 x m)
        softmax_jac = softmax_jacobian(yi)
        hidden_jac = previous_layer_jacobian(weights)
        first_layer_weights_jac = weight_matrix_jacobian(xi, (q, n))

        jac = entropy_jac.dot(softmax_jac).dot(hidden_jac).dot(first_layer_weights_jac)

        grad += jac

    return grad.reshape((q, n))


def model_gradients(x, hidden, loss, weights):
    """
    :param x:
    :param hidden:
    :param loss:
    :param weights:
    :return:
    """
    gradients = []
    # Negative signs are because we are doing gradient descent

    # First layer's gradient
    first_layer_grad = -previous_layers_weights_jacobian(x, weights, loss)
    gradients.append(first_layer_grad)

    # Last layer's gradient
    last_layer_grad = -softmax_layer_jacobian(hidden, loss)
    gradients.append(last_layer_grad)

    return gradients


def model_gradients_wrt_entropy(x, hidden, y, weights):
    gradients = []
    # No negative signs here because entropy is subject to maximization

    # First layer's gradient
    # TODO: This is not effecient, it does the same calculations twice
    first_layer_grad = first_layer_weights_gradient_wrt_entropy_loss(x, hidden, y, weights)
    gradients.append(first_layer_grad)

    # Last layer's gradient
    last_layer_grad = weights_gradient_wrt_entropy_loss(hidden, y)
    gradients.append(last_layer_grad)

    return gradients


def test_gradient_with_stacked_inputs():
    # Random seed
    np.random.seed(0)
    np.set_printoptions(precision=4, floatmode='fixed', linewidth=1000, suppress=True)

    layers = [4, 3, 2]
    model = create_model(layers)

    input_data = np.random.normal(0, 1, layers[0])
    output_data, hidden_data = policy_feed_forward(model, input_data)
    softmax_data = softmax(output_data)
    desired_label = [0, 1]

    # non_stacked_grad = model_gradients_wrt_entropy(input_data, hidden_data, softmax_data, model[1])
    # print("non_stacked_grad")
    # print(non_stacked_grad)

    input_data_stacked = np.vstack((input_data, input_data, input_data, input_data))
    softmax_data_stacked = np.vstack((softmax_data, softmax_data, softmax_data, softmax_data))
    hidden_data_stacked = np.vstack((hidden_data, hidden_data, hidden_data, hidden_data))
    desired_label_stacked = np.vstack((desired_label, desired_label, desired_label, desired_label))

    # gradients_stacked = model_gradients(input_data_stacked, hidden_data_stacked,
    #                                     softmax_data_stacked - desired_label_stacked, model[1])

    # Desired result (multiply by 4 because there are four stacked)
    # [array([[-0.0070,  0.0190,  0.0568, -0.0146],
    #         [ 0.0224, -0.0611, -0.1825,  0.0467],
    #         [-0.0106,  0.0289,  0.0863, -0.0221]]), array([[ 0.0372,  0.0499, -0.0090],
    #                                                        [-0.0372, -0.0499,  0.0090]])]
    entropy_grad_w_stack = model_gradients_wrt_entropy(input_data_stacked, hidden_data_stacked,
                                                       softmax_data_stacked, model[1])
    print("stacked")
    print(entropy_grad_w_stack)

    print()


def test_jacobians():
    # Random seed
    np.random.seed(0)

    layers = [3, 2]
    model = create_model(layers)

    input_data = np.random.normal(0, 1, layers[0])
    output_data = policy_feed_forward(model, input_data)
    softmax_data = softmax(output_data)
    desired_label = np.array([0, 1])
    error = desired_label - softmax_data

    weights_jacobian = weight_matrix_jacobian(input_data, model[0].shape)
    softmax_jac = softmax_jacobian(softmax_data)

    output_data_gradient = np.zeros_like(softmax_jac)
    for i in range(len(output_data)):
        dx = 0.01
        output_data_test = output_data.copy()
        output_data_test[i] += dx
        softmax_test = softmax(output_data_test)

        output_data_gradient[i] = (softmax_test - softmax_data) / dx

    assert np.allclose(output_data_gradient, softmax_jac, 0.01)


def test_entropy_loss():
    # Random seed
    np.random.seed(0)

    # Print arrays nicer
    np.set_printoptions(precision=4, floatmode='fixed', linewidth=1000, suppress=True)

    layers = [4, 3, 2]
    model = create_model(layers)

    print("model")
    print(model)

    input_data = np.random.normal(0, 1, layers[0])
    output_data, hidden_data = policy_feed_forward(model, input_data)
    softmax_data = softmax(output_data)
    entropy = entropy_of_probabilities(softmax_data)

    print("input_data")
    print(input_data)
    print("hidden_data")
    print(hidden_data)
    print("output_data")
    print(output_data)
    print("softmax_data")
    print(softmax_data)
    print("entropy")
    print(entropy)
    print()

    # Get gradients of first layer
    entropy_grad = model_gradients_wrt_entropy(input_data, hidden_data, softmax_data, model[1])

    # dx = 0.001
    # model[0][2, 1] += dx
    # output_data, _ = policy_feed_forward(model, input_data)
    # softmax_data = softmax(output_data)
    # entropy_modified = entropy_of_probabilities(softmax_data)
    # print("Test of gradient")
    # print((entropy_modified - entropy) / dx)

    weights = []
    learning_rate = 0.1
    entropy_bonus = 0.3
    desired_label = np.array([[1, 0]])
    print("New softmaxes:")
    for i in range(100):
        weights.append(model[0].copy().ravel())

        # entropy_grad = weights_gradient_wrt_entropy_loss(input_data, softmax_data)
        entropy_grad = model_gradients_wrt_entropy(input_data, hidden_data, softmax_data, model[1])

        # label_grad = model_gradients(input_data, hidden_data, desired_label - softmax_data, model[1])

        for i in range(len(model)):
            model[i] += (entropy_bonus * entropy_grad[i]) * learning_rate

        output_data, hidden_data = policy_feed_forward(model, input_data)
        softmax_data = softmax(output_data)

        print(softmax_data)

    import matplotlib.pyplot as plt
    plt.plot(np.vstack(weights))
    plt.show()


if __name__ == "__main__":
    # Test backpropagation
    test_gradient_with_stacked_inputs()

    # Test entropy bonus
    # test_entropy_loss()

    # # Random seed
    # np.random.seed(0)
    #
    # # Print arrays nicer
    # np.set_printoptions(precision=4, floatmode='fixed', linewidth=1000, suppress=True)
    #
    # layers = [4, 3, 2]
    # model = create_model(layers)
    #
    # print("model")
    # print(model)
    #
    # input_data = np.random.normal(0, 1, layers[0])
    # output_data, hidden_data = policy_feed_forward(model, input_data)
    # softmax_data = softmax(output_data)
    # desired_label = np.array([0, 1])
    # error = desired_label - softmax_data
    #
    # print("input data")
    # print(input_data)
    # print("hidden data")
    # print(hidden_data)
    # print("output data")
    # print(output_data)
    # print("softmax")
    # print(softmax_data)
    # print("desired label")
    # print(desired_label)
    # print("error")
    # print(error)
    # print()
    #
    # # TODO: These need to be all stacked
    # # TODO: subtract label and softmax before passing to function
    # output_data, hidden_data = policy_feed_forward(model, input_data)
    # softmax_data = softmax(output_data)
    #
    # simple_layer_jac = softmax_layer_jacobian(hidden_data, softmax_data, desired_label)
    # print(simple_layer_jac)
    #
    # jac_wrt_inputs = softmax_layer_jacobian_wrt_inputs(model[1], softmax_data, desired_label)
    # print("jac_wrt_inputs")
    # print(jac_wrt_inputs)
    # dx = 0.01
    # # hidden_data[0] += dx
    # # softmax_data_modified = softmax(model[1].dot(hidden_data))
    # prev_layer_grad = previous_layers_weights_jacobian(input_data, model[1], softmax_data, desired_label)
    # print("prev_layer_grad")
    # print(prev_layer_grad)
    #
    # gradients = model_gradients(input_data, hidden_data, softmax_data, model[1], desired_label)
    # print("gradients")
    # print(gradients)
    #
    # model[0][1, 2] += dx
    # output_data, _ = policy_feed_forward(model, input_data)
    # softmax_modified = softmax(output_data)
    # print(softmax_modified)
    # print(softmax_data)
    # # print((softmax_modified - softmax_data) / dx)
    #
    # # weights = []
    # # for i in range(30):
    # #     weights.append(model[0].copy().ravel())
    # #
    # #     model = backpropagation(model, input_data, softmax_data, desired_label, learning_rate=1)
    # #     output_data_modified = policy_feed_forward(model, input_data)
    # #     softmax_data = softmax(output_data_modified)
    # #     print("new softmax")
    # #     print(softmax_data)
    # #
    # # print(np.vstack(weights))
    # #
    # # import matplotlib.pyplot as plt
    # # plt.plot(np.vstack(weights))
    # # plt.show()
