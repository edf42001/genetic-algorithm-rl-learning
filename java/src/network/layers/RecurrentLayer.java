package network.layers;

import network.math.Activations;
import network.math.Matrix;

/**
 * A simple recurrent layer where the entire middle state is fed back into itself
 */
public class RecurrentLayer extends Layer
{
    private Matrix weights;
    private Matrix biases;

    private Matrix hiddenState;


    public RecurrentLayer(int inputSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.randomizeBiases();
        this.randomizeWeights();

        //Initialize to all 0s
        this.hiddenState = new Matrix(1, outputSize);
    }

    public RecurrentLayer(Matrix weights, Matrix biases)
    {
        this.inputSize = weights.getRows();
        this.outputSize = weights.getCols();

        this.weights = weights;
        this.biases = biases;

        //Initialize to all 0s
        this.hiddenState = new Matrix(1, outputSize);
    }

    public void randomizeBiases() {
        // Just set these to uniformly distributed in [-0.5, 0.5]
        this.biases = Matrix.randomUniform(1, this.outputSize, 0.5f);
    }

    public void randomizeWeights() {
        // This range value is called the glorot_uniform initialization
        float range = (float) Math.sqrt(6.0 / (this.inputSize + 2 * this.outputSize));
        this.weights = Matrix.randomUniform(this.inputSize + this.outputSize, this.outputSize, range);
    }

    @Override
    public Matrix feedForward(Matrix input) {
        // Combine input and hidden state
        Matrix combinedInOut = input.concatenateRow(hiddenState);

        hiddenState = combinedInOut.dot(weights).add(biases);

        // Use tanh activation (modifies in place)
        Activations.applyActivation(hiddenState, 3);
        return hiddenState;
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
    public RecurrentLayer fromLargerArray(float[] arr, int index) {
        Matrix newWeights = new Matrix(weights.getRows(), weights.getCols());
        Matrix newBiases  = new Matrix(biases.getRows(), biases.getCols());

        Matrix.loadWeightsBiasesFromArray(arr, index, newWeights, newBiases);

        return new RecurrentLayer(newWeights, newBiases);
    }

    @Override
    public Layer clone() {
        return new RecurrentLayer(weights.clone(), biases.clone());
    }

    @Override
    public Matrix getWeights() {
        return weights;
    }

    @Override
    public Matrix getBiases() {
        return biases;
    }

    @Override
    public int numParams() {
        return weights.numParams() + biases.numParams();
    }
}
