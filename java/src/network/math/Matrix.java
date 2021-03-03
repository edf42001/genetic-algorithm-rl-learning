package network.math;

import java.io.Serializable;

public class Matrix implements Serializable {

    // local variables
    private int rows;
    private int cols;
    private float[][] data;

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //constructor
    public Matrix(int r, int c) {
        rows = r;
        cols = c;
        data = new float[rows][cols];
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //constructor from 2D array
    public Matrix(float[][] m) {
        data = m;
        rows = m.length;
        cols = m[0].length;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //print matrix
    public void print() {
        System.out.println(this);
    }

    //Convert to string
    public String toString() {
        String ret = "";
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                ret += String.format("%.2f", data[r][c]);
                if (c != cols - 1) {
                    ret += " ";
                }
            }
            ret += "\n";
        }
        return ret;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    //multiply by scalar
    public Matrix multiply(float n) {
        Matrix newMatrix = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newMatrix.data[i][j] = data[i][j] * n;
            }
        }
        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //return a matrix which is this matrix dot product parameter matrix
    public Matrix dot(Matrix n) {
        Matrix result = new Matrix(rows, n.cols);

        if (cols != n.rows) {
            throw new IndexOutOfBoundsException(String.format("Matrix cols do not equal rows for dot: %d, %d", cols, n.rows));
        }

        //for each spot in the new matrix
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < n.cols; j++) {
                float sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * n.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //set the matrix to random floats between -1 and 1
    public static Matrix randomUniform(int rows, int cols, float range) {
        Matrix ret = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                ret.data[i][j] = range * (2 * MyRand.randFloat() - 1);
            }
        }

        return ret;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //add a scalar to the matrix
    public Matrix add(float n) {
        Matrix newMatrix = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newMatrix.data[i][j] = data[i][j] + n;
            }
        }

        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    ///return a matrix which is this matrix + parameter matrix
    public Matrix add(Matrix n) {
        Matrix newMatrix = new Matrix(rows, cols);

        if (cols != n.cols || rows != n.rows) {
            throw new IndexOutOfBoundsException(String.format("Matrix cols do not equal rows for add: %d, %d, %d, %d",
                    rows, n.rows, cols, n.cols));
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newMatrix.data[i][j] = data[i][j] + n.data[i][j];
            }
        }

        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Subtracts this matrix from a scalar
    public Matrix subtractFrom(float n) {
        Matrix newMatrix = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newMatrix.data[i][j] = n - data[i][j];
            }
        }

        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //return a matrix which is this matrix - parameter matrix
    public Matrix subtract(Matrix n) {
        Matrix newMatrix = new Matrix(cols, rows);
        if (cols == n.cols && rows == n.rows) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    newMatrix.data[i][j] = data[i][j] - n.data[i][j];
                }
            }
        }
        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //return a matrix which is this matrix * parameter matrix (element wise multiplication)
    public Matrix pointwiseMultiply(Matrix n) {
        Matrix newMatrix = new Matrix(rows, cols);

        if (cols == n.cols && rows == n.rows) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    newMatrix.data[i][j] = data[i][j] * n.data[i][j];
                }
            }
        }

        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //return a matrix which is the transpose of this matrix
    public Matrix transpose() {
        Matrix n = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                n.data[j][i] = data[i][j];
            }
        }
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Concatenates two 1 x N row matrices together
    public Matrix concatenateRow(Matrix n) {
        Matrix newMatrix = new Matrix(1, cols + n.cols);

        for (int i = 0; i < cols; i++) {
            newMatrix.data[0][i] = data[0][i];
        }

        for (int i = cols; i < cols + n.cols; i++) {
            newMatrix.data[0][i] = n.data[0][i - cols];
        }

        return newMatrix;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //Creates a single column array from the parameter array
    public static Matrix singleColumnMatrixFromArray(float[] arr) {
        Matrix n = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            n.data[i][0] = arr[i];
        }
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //sets this matrix from an array
    public void fromArray(float[] arr) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = arr[j + i * cols];
            }
        }
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //for ix1 matrixes adds one to the bottom
    public Matrix addBias() {
        Matrix n = new Matrix(rows + 1, 1);
        for (int i = 0; i < rows; i++) {
            n.data[i][0] = data[i][0];
        }
        n.data[rows][0] = 1;
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //returns the matrix that is the derived sigmoid function of the current matrix
    public Matrix sigmoidDerived() {
        Matrix n = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                n.data[i][j] = (data[i][j] * (1 - data[i][j]));
            }
        }
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //returns the matrix which is this matrix with the bottom layer removed
    public Matrix removeBottomLayer() {
        Matrix n = new Matrix(rows - 1, cols);
        for (int i = 0; i < n.rows; i++) {
            for (int j = 0; j < cols; j++) {
                n.data[i][j] = data[i][j];
            }
        }
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //Mutation function for genetic algorithm
    public void mutate(float mutationRate, float mutationSize) {
        //for each element in the matrix
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                //if chosen to be mutated
                if (MyRand.randFloat() < mutationRate) {
                    data[i][j] += 2 * mutationSize * (MyRand.randFloat() - 0.5);

                    //TODO implement bounds on mutation?
//					//set the boundaries to 1 and -1
//					if (matrix[i][j]>1) {
//						matrix[i][j] = 1;
//					}
//					if (matrix[i][j] <-1) {
//						matrix[i][j] = -1;
//					}
                }
            }
        }
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //returns a matrix which has a random number of values from this matrix and the rest from the parameter matrix
    public Matrix crossover(Matrix partner) {
        Matrix child = new Matrix(rows, cols);

        //pick a random point in the matrix
        int randValue = MyRand.randInt(rows * cols);
        int randR = randValue / cols; // Round down
        int randC = randValue % cols; // Find column

//        System.out.println(randValue + " " + randR + " " + randC);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if ((i < randR) || (i == randR && j <= randC)) {
                    // If before the random point then copy from this matrix
                    child.data[i][j] = data[i][j];
                } else {
                    //if after the random point then copy from the parameter array
                    child.data[i][j] = partner.data[i][j];
                }
            }
        }

        return child;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Combines a weights and biases matrix into a larger flat array
    public static void insertWeightsBiasesIntoArray(float[] arr, int index, Matrix weights, Matrix biases) {
        int increment = index;

        int rows = weights.getRows();
        int cols = weights.getCols();

        // Interleaves the weights and biases
        // One row of weights, then the associated bias
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                arr[increment + r] = weights.data[r][c];
            }
            increment += rows;
            arr[increment] = biases.data[0][c];
            increment += 1;
        }
    }

    // Loads data from an interleaved weights biases array into a weights and a biases matrix
    public static void loadWeightsBiasesFromArray(float[] arr, int index, Matrix weights, Matrix biases) {
        int increment = index;

        int rows = weights.getRows();
        int cols = weights.getCols();

        // LoInterleaves the weights and biases
        // One row of weights, then the associated bias
        for (int c = 0; c < cols; c++)
        {
            for (int r = 0; r < rows; r++)
            {
                weights.data[r][c] = arr[increment + r];
            }
            increment += rows;
            biases.data[0][c] = arr[increment];
            increment += 1;
        }
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Inserts this matrix into an array
    public static void insertIntoArray(float[] arr, int index, Matrix mat) {
        int increment = index;

        int rows = mat.getRows();
        int cols = mat.getCols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[increment + j] = mat.data[i][j];
            }
            increment += cols;
        }
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //return a copy of this matrix
    public Matrix clone() {
        Matrix clone = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clone.data[i][j] = data[i][j];
            }
        }
        return clone;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    // Number of values this matrix has
    public int numParams()
    {
        return rows * cols;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //Get rows
    public int getRows() {
        return rows;
    }

    //Get columns
    public int getCols() {
        return cols;
    }

    public float[][] getData() {
        return data;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    public static void insertInto(float[] big, float[] small, int index) {
        for (int i = index; i < index + small.length && i < big.length; i++) {
            big[i] = small[i - index];
        }
    }

    public static float[] subset(float[] arr, int index, int length) {
        float[] ret = new float[length];
        for (int i = 0; i < length; i++) {
            ret[i] = arr[i + index];
        }
        return ret;
    }
}
