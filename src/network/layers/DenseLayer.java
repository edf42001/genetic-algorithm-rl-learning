package network.layers;

import network.math.Matrix;

/***
A fully connected layer in a
 neural network

 ***/
public class DenseLayer extends Layer {
    private Matrix weights;
    private Matrix biases;

    private int activation;

    public DenseLayer(int inputSize, int outputSize, String activation)
    {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.randomizeWeights();
        this.randomizeBiases();

        // Set activation types as ints for
        // faster comparison
        this.activation = stringToActivation(activation);

    }

    public DenseLayer(Matrix weights, Matrix biases, int activation) {
        this.weights = weights;
        this.biases = biases;

        this.inputSize = weights.getRows();
        this.outputSize = weights.getCols();

        this.activation = activation;
    }

    public void randomizeBiases() {
        // Just set these to uniformly distributed in [-0.5, 0.5]
        this.biases = Matrix.randomUniform(1, this.outputSize, 0.5f);
    }

    public void randomizeWeights() {
        // This range value is called the glorot_uniform initialization
        float range = (float) Math.sqrt(6.0 / (this.inputSize + this.outputSize));
        this.weights = Matrix.randomUniform(this.inputSize, this.outputSize, range);
    }

    @Override
    public Matrix feedForward(Matrix input)
    {
        // Order of multiply is due to choosing layers to be row vectors
        // Instead of col vectors
        Matrix out = input.dot(weights).add(biases);

        if (activation == 0) {
            // relu
            return DenseLayer.relu(out);
        }
        else if (activation == 1) {
            // sigmoid
            return DenseLayer.sigmoid(out);
        }
        else if (activation == 2) {
            // none
            return out;
        }
        System.err.println("Error: Unknown activation " + activation);
        return out;
    }

    @Override
    public void mutate(float mutationRate, float mutationSize) {
        weights.mutate(mutationRate, mutationSize);
        biases.mutate(mutationRate, mutationSize);
    }

    public static Matrix relu(Matrix a)
    {
        float[][] data = a.getData();

        // Apply relu function to a matrix
        // This assumes the matrix is a row matrix
        for (int i = 0; i < data[0].length; i++)
        {
            if (data[0][i] < 0) {
                data[0][i] = 0;
            }
        }

        return a;
    }

    public static Matrix sigmoid(Matrix a)
    {
        float[][] data = a.getData();

        // Apply rele function to a matrix
        // This assumes the matrix is a row matrix
        for (int i = 0; i < data[0].length; i++)
        {
            data[0][i] = (float) (1.0 / (1.0 + Math.exp(-data[0][i])));
        }

        return a;
    }

    public int stringToActivation(String activation)
    {
        switch (activation) {
            case "relu":
                return 0;
            case "sigmoid":
                return 1;
            case "none":
                return 2;
            default:
                System.err.println("Error: Unknown activation " + activation);
                return -1;
        }
    }

    public DenseLayer clone()
    {
        return new DenseLayer(weights.clone(), biases.clone(), activation);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public int getActivation() {
        return activation;
    }
}
