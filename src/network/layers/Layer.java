package network.layers;

import network.math.Matrix;

import java.io.Serializable;

public abstract class Layer implements Serializable {
    // Input and output size of network
    protected int inputSize;
    protected int outputSize;

    /**
     * Feed data through this layer to get an output
     * @param input The input data to this layer
     * @return The output of this layer
     */
    public abstract Matrix feedForward(Matrix input);

//    public abstract void mutate(float rate);
//
//    public abstract Layer crossover(Layer partner);

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public abstract void mutate(float mutationRate, float mutationSize);

    public abstract Layer crossover(Layer other);

    public abstract Layer clone();

    public abstract Matrix getWeights();

    public abstract Matrix getBiases();
}
