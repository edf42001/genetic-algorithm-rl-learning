import java.io.Serializable;

/***
A fully connected layer in a
 neural network

 ***/
public class DenseLayer implements Serializable {
    private int inputSize;
    private int outputSize;

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
        switch (activation) {
            case "relu":
                this.activation = 0;
                break;
            case "sigmoid":
                this.activation = 1;
                break;
            case "none":
                this.activation = 2;
                break;
            default:
                System.err.println("Error: Unknown activation " + activation);
                break;
        }
    }

    public void randomizeBiases() {
        this.biases = Matrix.randomMatrix(1, this.outputSize, 0, 0.5f);
    }

    public void randomizeWeights() {
        this.weights = Matrix.randomMatrix(this.inputSize, this.outputSize, 0, 0.5f);
    }

    public Matrix feedForward(Matrix inputData)
    {
        // Order of multiply is due to choosing layers to be row vectors
        // Instead of col vectors
        Matrix out = Matrix.add(Matrix.multiply(inputData, weights), biases);

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

    public static Matrix relu(Matrix a)
    {
        float[][] data = a.getData();

        // Apply rele function to a matrix
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

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }


}
