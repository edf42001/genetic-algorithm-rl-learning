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


def softmax_layer_jacobian(x, y, label):
    """
    :param x: The inputs to this layer (batch_size x n)
    :param y: The softmax outputs from this layer (batch_size x m)
    Calculates the jacobian operator with respect to weights on the output cross entropy loss, but with a simpler method
    :return:
    """

    return np.dot((y - label).T, x)


def softmax_layer_jacobian_wrt_inputs(weights, y, label):
    # Equivalent to:
    # softmax_jac = softmax_jacobian(y)
    # cross_entropy_jac = cross_entropy_jacobian(y, label)
    # return cross_entropy_jac.dot(softmax_jac.dot(weights))

    # (batch x m), (m X h) -> (batch_size x h)
    return np.dot(y - label, weights)


def previous_layers_weights_jacobian(x, weights, y, label):
    """
    :param x: (batch size x n)
    :param weights: (m x h)
    :param y: (batch_size x m)
    :param label: (batch_size x m)
    :return: previous layers weights (h x n)
    """

    # d_hidden: gradient of hidden layer, (batch_size x h)
    d_hidden = softmax_layer_jacobian_wrt_inputs(weights, y, label)

    return np.dot(d_hidden.T, x)


def model_gradients(x, hidden, y, weights, label):
    """
    :param x:
    :param hidden:
    :param y:
    :param weights:
    :param label:
    :return:
    """
    gradients = []
    # Negative signs are because we are doing gradient descent

    # First layer's gradient
    first_layer_grad = -previous_layers_weights_jacobian(x, weights, y, label)
    gradients.append(first_layer_grad)

    # Last layer's gradient
    last_layer_grad = -softmax_layer_jacobian(hidden, y, label)
    gradients.append(last_layer_grad)

    print("first_layer_grad")
    print(first_layer_grad)

    print("last_layer_grad")
    print(last_layer_grad)

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

    # gradients_non_stacked = model_gradients(input_data, hidden_data, softmax_data, model[1], desired_label)
    # print(gradients_non_stacked)

    input_data_stacked = np.vstack((input_data, input_data, input_data, input_data))
    softmax_data_stacked = np.vstack((softmax_data, softmax_data, softmax_data, softmax_data))
    hidden_data_stacked = np.vstack((hidden_data, hidden_data, hidden_data, hidden_data))
    desired_label_stacked = np.vstack((desired_label, desired_label, desired_label, desired_label))

    gradients_stacked = model_gradients(input_data_stacked, hidden_data_stacked, softmax_data_stacked, model[1], desired_label_stacked)
    print("stacked")
    print(gradients_stacked)

    # [array([[-0.0458,  0.1248,  0.3731, -0.0955],
    #           [ 0.1469, -0.4009, -1.1982,  0.3068],
    #           [-0.0695,  0.1896,  0.5667, -0.1451]]),
    #           array([[ 0.2439,  0.3273, -0.0592],
    #                  [-0.2439, -0.3273,  0.0592]])]


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


if __name__ == "__main__":
    # Test backpropagation
    test_gradient_with_stacked_inputs()

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
    desired_label = np.array([0, 1])
    error = desired_label - softmax_data

    print("input data")
    print(input_data)
    print("hidden data")
    print(hidden_data)
    print("output data")
    print(output_data)
    print("softmax")
    print(softmax_data)
    print("desired label")
    print(desired_label)
    print("error")
    print(error)
    print()

    # TODO: These need to be all stacked
    # TODO: subtract label and softmax before passing to function
    output_data, hidden_data = policy_feed_forward(model, input_data)
    softmax_data = softmax(output_data)

    simple_layer_jac = softmax_layer_jacobian(hidden_data, softmax_data, desired_label)
    print(simple_layer_jac)

    jac_wrt_inputs = softmax_layer_jacobian_wrt_inputs(model[1], softmax_data, desired_label)
    print("jac_wrt_inputs")
    print(jac_wrt_inputs)
    dx = 0.01
    # hidden_data[0] += dx
    # softmax_data_modified = softmax(model[1].dot(hidden_data))
    prev_layer_grad = previous_layers_weights_jacobian(input_data, model[1], softmax_data, desired_label)
    print("prev_layer_grad")
    print(prev_layer_grad)

    gradients = model_gradients(input_data, hidden_data, softmax_data, model[1], desired_label)
    print("gradients")
    print(gradients)

    model[0][1, 2] += dx
    output_data, _ = policy_feed_forward(model, input_data)
    softmax_modified = softmax(output_data)
    print(softmax_modified)
    print(softmax_data)
    # print((softmax_modified - softmax_data) / dx)

    # weights = []
    # for i in range(30):
    #     weights.append(model[0].copy().ravel())
    #
    #     model = backpropagation(model, input_data, softmax_data, desired_label, learning_rate=1)
    #     output_data_modified = policy_feed_forward(model, input_data)
    #     softmax_data = softmax(output_data_modified)
    #     print("new softmax")
    #     print(softmax_data)
    #
    # print(np.vstack(weights))
    #
    # import matplotlib.pyplot as plt
    # plt.plot(np.vstack(weights))
    # plt.show()



