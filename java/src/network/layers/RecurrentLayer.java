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
    private Matrix hiddenWeights;

    private Matrix hiddenState;


    public RecurrentLayer(int inputSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.randomizeBiases();
        this.randomizeWeights();
        this.randomizeHiddenWeights();

        //Initialize to all 0s
        this.hiddenState = new Matrix(1, outputSize);
    }

    public RecurrentLayer(Matrix weights, Matrix biases, Matrix hiddenWeights)
    {
        this.inputSize = weights.getRows();
        this.outputSize = weights.getCols();

        this.weights = weights;
        this.biases = biases;
        this.hiddenWeights = hiddenWeights;

        //Initialize to all 0s
        this.hiddenState = new Matrix(1, outputSize);
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

    public void randomizeHiddenWeights()
    {
        // This range value is called the glorot_uniform initialization
        float range = (float) Math.sqrt(6.0 / (2 * this.outputSize));
        this.hiddenWeights = Matrix.randomUniform(this.outputSize, this.outputSize, range);
    }

    @Override
    public Matrix feedForward(Matrix input) {
        hiddenState = input.dot(weights).add(hiddenState.dot(hiddenWeights)).add(biases);
        // Use tanh activation
        Activations.applyActivation(hiddenState, 3);
        return hiddenState;
    }

    @Override
    public void mutate(float mutationRate, float mutationSize) {
        weights.mutate(mutationRate, mutationSize);
        biases.mutate(mutationRate, mutationSize);
        hiddenWeights.mutate(mutationRate, mutationSize);
    }

    @Override
    public Layer crossover(Layer other) {
        RecurrentLayer layer = (RecurrentLayer) other;
        Matrix newWeights = weights.crossover(layer.weights);
        Matrix newBiases = biases.crossover(layer.biases);
        Matrix newHiddenWeights = hiddenWeights.crossover(layer.hiddenWeights);

        return new RecurrentLayer(newWeights, newBiases, newHiddenWeights);
    }

    @Override
    public Layer clone() {
        return new RecurrentLayer(weights.clone(), biases.clone(), hiddenWeights.clone());
    }

    @Override
    public Matrix getWeights() {
        return weights;
    }

    @Override
    public Matrix getBiases() {
        return biases;
    }
}
