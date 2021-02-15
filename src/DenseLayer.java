/***
A fully connected layer in a
 neural network

 ***/
public class DenseLayer {
    private int inputSize;
    private int outputSize;

    private Matrix weights;
    private Matrix biases;

    public DenseLayer(int inputSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.randomizeWeights();
        this.randomizeBiases();
    }

    public void randomizeBiases() {
        this.biases = Matrix.randomMatrix(1, this.outputSize);
    }

    public void randomizeWeights() {
        this.weights = Matrix.randomMatrix(this.inputSize, this.outputSize);
    }

    public Matrix feedForward(Matrix inputData)
    {
        // Order of multiply is due to choosing layers to be row vectors
        // Instead of col vectors
        return Matrix.add(Matrix.multiply(inputData, weights), biases);
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }


}
