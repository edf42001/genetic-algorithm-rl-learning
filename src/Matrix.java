import java.io.Serializable;
import java.util.Random;

public class Matrix implements Serializable {
    public static Random random = new Random();

    private float[][] data;

    public Matrix(int rows, int cols){
        this.data = new float[rows][cols];
    }

    public Matrix(float[][] data)
    {
        this.data = data;
    }

    public void setData(float[][] data)
    {
        this.data = data.clone();
        //TODO test pass by value
    }

    public static Matrix add(Matrix a, Matrix b)
    {
        int rows = a.data.length;
        int cols = a.data[0].length;

        float[][] sum = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                sum[r][c] = a.data[r][c] + b.data[r][c];
            }
        }

        return new Matrix(sum);
    }

    public static Matrix subtract(Matrix a, Matrix b)
    {
        int rows = a.data.length;
        int cols = a.data[0].length;

        float[][] sum = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                sum[r][c] = a.data[r][c] - b.data[r][c];
            }
        }

        return new Matrix(sum);
    }

    public static Matrix multiply(Matrix a, Matrix b)
    {
        int rows1 = a.data.length;
        int cols2 = b.data[0].length;
        int rows2 = b.data.length;

        float[][] ret = new float[rows1][cols2];

        for (int r = 0; r < rows1; r++) {
            for (int c = 0; c < cols2; c++) {
                for (int k = 0; k < rows2; k++) {
                    ret[r][c] += a.data[r][k] * b.data[k][c];
                }
            }
        }

        return new Matrix(ret);
    }

    public static Matrix randomMatrix(int rows, int cols, float mean, float std)
    {
        float[][] ret = new float[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                ret[r][c] = (float) (std * (random.nextGaussian() - mean));
            }
        }

        return new Matrix(ret);
    }

    public static Matrix randomMatrix(int rows, int cols)
    {
        return Matrix.randomMatrix(rows, cols, 0, 1);
    }

    public String toString()
    {
        String ret = "";

        int rows = data.length;
        int cols = data[0].length;

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

    public float[][] getData() { return this.data; }

}
