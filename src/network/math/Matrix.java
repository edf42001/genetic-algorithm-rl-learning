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
    public Matrix multiply(Matrix n) {
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
    //returns an array which represents this matrix
    public float[] toArray() {
        float[] arr = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[j + i * cols] = data[i][j];
            }
        }
        return arr;
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
    //applies the activation function(sigmoid) to each element of the matrix
    public Matrix activate() {
        Matrix n = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                n.data[i][j] = sigmoid(data[i][j]);
            }
        }
        return n;
    }

    //---------------------------------------------------------------------------------------------------------------------------------------------------------
    //sigmoid activation function
    float sigmoid(float x) {
        return (float) (1.0 / (1 + Math.exp(-x)));
    }

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
        int randC = (int) Math.floor(MyRand.randInt(cols));
        int randR = (int) Math.floor(MyRand.randInt(rows));

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
//
//public class Matrix implements Serializable {
//    public static Random random = new Random();
//
//    private float[][] data;
//
//    public Matrix(int rows, int cols){
//        this.data = new float[rows][cols];
//    }
//
//    public Matrix(float[][] data)
//    {
//        this.data = data;
//    }
//
//    public void setData(float[][] data)
//    {
//        this.data = data.clone();
//        //TODO test pass by value
//    }
//
//    public static Matrix add(Matrix a, Matrix b)
//    {
//        int rows = a.data.length;
//        int cols = a.data[0].length;
//
//        float[][] sum = new float[rows][cols];
//
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                sum[r][c] = a.data[r][c] + b.data[r][c];
//            }
//        }
//
//        return new Matrix(sum);
//    }
//
//    public static Matrix subtract(Matrix a, Matrix b)
//    {
//        int rows = a.data.length;
//        int cols = a.data[0].length;
//
//        float[][] sum = new float[rows][cols];
//
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                sum[r][c] = a.data[r][c] - b.data[r][c];
//            }
//        }
//
//        return new Matrix(sum);
//    }
//
//    public static Matrix multiply(Matrix a, Matrix b)
//    {
//        int rows1 = a.data.length;
//        int cols2 = b.data[0].length;
//        int rows2 = b.data.length;
//
//        float[][] ret = new float[rows1][cols2];
//
//        for (int r = 0; r < rows1; r++) {
//            for (int c = 0; c < cols2; c++) {
//                for (int k = 0; k < rows2; k++) {
//                    ret[r][c] += a.data[r][k] * b.data[k][c];
//                }
//            }
//        }
//
//        return new Matrix(ret);
//    }
//
//    public static Matrix multiply(float a, Matrix b)
//    {
//        int rows = b.data.length;
//        int cols = b.data[0].length;
//
//        float[][] ret = new float[rows][cols];
//
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                ret[r][c] = a * b.data[r][c];
//            }
//        }
//
//        return new Matrix(ret);
//    }
//    /**
//     * Return a matrix with uniformly distributed
//     * random values in each entry
//     * @param rows Rows
//     * @param cols Cols
//     * @param range Values will be in [-range, range]
//     * @return
//     */
//    public static Matrix randomMatrix(int rows, int cols, float range)
//    {
//        float[][] ret = new float[rows][cols];
//
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                ret[r][c] = 2 * range * (MyRand.randFloat() - 0.5f);
//            }
//        }
//
//        return new Matrix(ret);
//    }
//
//    public static Matrix randomMatrix(int rows, int cols)
//    {
//        return Matrix.randomMatrix(rows, cols, 1);
//    }
//
//    public String toString()
//    {
//        String ret = "";
//
//        int rows = data.length;
//        int cols = data[0].length;
//
//        for (int r = 0; r < rows; r++) {
//            for (int c = 0; c < cols; c++) {
//                ret += String.format("%.2f", data[r][c]);
//                if (c != cols - 1) {
//                    ret += " ";
//                }
//            }
//            ret += "\n";
//        }
//
//        return ret;
//    }
//
//    public Matrix clone()
//    {
//        // Start size determined by data and doesn't matter
//        Matrix ret = new Matrix(0,0);
//        ret.data = data.clone();
//        return ret;
//    }
//
//    public float[][] getData() { return this.data; }
//
//}
