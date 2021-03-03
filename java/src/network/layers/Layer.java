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

//    public abstract Layer naiveCrossover(Layer other);

    /**
     * Flattens this layer and inserts it into a bigger array at
     * as specific index
     * @param arr Array to insert into
     * @param index Index to insert at
     */
    public abstract void insertIntoArray(float[] arr, int index);

    /**
     * Generates the parameters for this layer from a section
     * of a larger array. This is how the chromosome gets converted
     * back into a layer
     * @param arr Array we are creating this layer from
     * @param index Index to take data from
     */
    public abstract Layer fromLargerArray(float[] arr, int index);

    public abstract Layer clone();

    public abstract Matrix getWeights();

    public abstract Matrix getBiases();

    /**
     * @return Number of parameters in this layer
     */
    public abstract int numParams();
}
