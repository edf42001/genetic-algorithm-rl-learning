import java.util.Random;

public class Matrix {
    public static Random random = new Random(0);

    private double[][] data;

    public Matrix(int rows, int cols){
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data)
    {
        this.data = data;
    }

    public void setData(double[][] data)
    {
        this.data = data.clone();
        //TODO test pass by value
    }

    public static Matrix add(Matrix a, Matrix b)
    {
        int rows = a.data.length;
        int cols = a.data[0].length;

        double[][] sum = new double[rows][cols];

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

        double[][] sum = new double[rows][cols];

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

        double[][] ret = new double[rows1][cols2];

        for (int r = 0; r < rows1; r++) {
            for (int c = 0; c < cols2; c++) {
                for (int k = 0; k < rows2; k++) {
                    ret[r][c] += a.data[r][k] * b.data[k][c];
                }
            }
        }

        return new Matrix(ret);
    }

    public static Matrix randomMatrix(int rows, int cols, double mean, double std)
    {
        double[][] ret = new double[rows][cols];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                ret[r][c] = random.nextGaussian();
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

    public double[][] getData() { return this.data; }

}
