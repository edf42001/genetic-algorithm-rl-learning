public class Main {

    public static void main(String[] args) {
        DenseLayer layer = new DenseLayer(3, 2);

        layer.randomizeWeights();
        layer.randomizeBiases();

        System.out.println(layer.getWeights());
        System.out.println(layer.getBiases());

        Matrix input = new Matrix(1, 3);
        input.setData(new double[][] {{1, 2, 3}});

        Matrix output = layer.feedForward(input);
        System.out.println(output);
    }

    public static void testMatrix()
    {
        Matrix a = new Matrix(2, 2);
        Matrix b = new Matrix(2, 2);

        a.setData(new double[][] {{1 , 2}});
        b.setData(new double[][] {{-1, 0}, {1, 2}});

        System.out.println(a);
        System.out.println(b);
//        System.out.println(Matrix.add(a, b));
        System.out.println(Matrix.multiply(a, b));
//        System.out.println(Matrix.randomMatrix(2, 2));
    }
}
