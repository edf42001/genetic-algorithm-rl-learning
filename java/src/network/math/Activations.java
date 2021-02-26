package network.math;

/**
 * A class for different output activations.
 * Assumes all matrices are output weights
 * and are 1 x N
 */
public class Activations {
    public static void sigmoid(Matrix mat)
    {
        float[] data = mat.getData()[0];
        for (int i = 0; i < mat.getCols(); i++) {
           data[i] = sigmoid(data[i]);
        }
    }

    public static void relu(Matrix mat)
    {
        float[] data = mat.getData()[0];
        for (int i = 0; i < mat.getCols(); i++) {
            data[i] = relu(data[i]);
        }
    }

    public static void tanh(Matrix mat)
    {
        float[] data = mat.getData()[0];
        for (int i = 0; i < mat.getCols(); i++) {
            data[i] = tanh(data[i]);
        }
    }

    public static float sigmoid(float x)
    {
        return (float) (1.0 / (1 + Math.exp(-x)));
    }

    public static float relu(float x)
    {
        if (x > 0)
        {
            return x;
        }
        return 0;
    }

    public static float tanh(float x)
    {
        return (float) Math.tanh(x);
    }

    public static int stringToActivation(String activation)
    {
        switch (activation) {
            case "relu":
                return 0;
            case "sigmoid":
                return 1;
            case "none":
                return 2;
            case "tanh":
                return 3;
            default:
                System.err.println("Error: Unknown activation " + activation);
                return -1;
        }
    }

    public static void applyActivation(Matrix out, int activation) {
        if (activation == 0) {
            // relu
            Activations.relu(out);
        } else if (activation == 1) {
            // sigmoid
            Activations.sigmoid(out);
        } else if (activation == 2) {
            // none
        } else if (activation == 3) {
            // tanh
            Activations.tanh(out);
        } else {
            System.err.println("Error: Unknown activation " + activation);
        }
    }
}
