package network.layers;

import network.Network;
import network.math.Activations;
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
        this.activation = Activations.stringToActivation(activation);

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
        Activations.applyActivation(out, this.activation);
        return out;
    }

    @Override
    public void mutate(float mutationRate, float mutationSize) {
        weights.mutate(mutationRate, mutationSize);
        biases.mutate(mutationRate, mutationSize);
    }

    @Override
    public void insertIntoArray(float[] arr, int index) {
        Matrix.insertWeightsBiasesIntoArray(arr, index, weights, biases);
    }

    @Override
    public DenseLayer fromLargerArray(float[] arr, int index) {
        Matrix newWeights = new Matrix(weights.getRows(), weights.getCols());
        Matrix newBiases  = new Matrix(biases.getRows(), biases.getCols());

        Matrix.loadWeightsBiasesFromArray(arr, index, newWeights, newBiases);

        return new DenseLayer(newWeights, newBiases, activation);
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

    @Override
    public int numParams() {
        return weights.numParams() + biases.numParams();
    }

    public int getActivation() {
        return activation;
    }
}
