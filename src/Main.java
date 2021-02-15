import sun.nio.ch.Net;

public class Main {

    public static void main(String[] args) {
        DenseLayer layer = new DenseLayer(3, 2);
        layer.randomizeWeights();
        layer.randomizeBiases();

        DenseLayer layer2 = new DenseLayer(2, 1);
        layer2.randomizeWeights();
        layer2.randomizeBiases();

        Network network = new Network();
        network.addLayer(layer);
        network.addLayer(layer2);

        System.out.println(layer.getWeights());
        System.out.println(layer.getBiases());

        System.out.println(layer2.getWeights());
        System.out.println(layer2.getBiases());

        Matrix input = new Matrix(1, 3);
        input.setData(new float[][] {{1, 2, 3}});

        Matrix output = layer.feedForward(input);
        System.out.println(output);

        Matrix output2 = network.feedForward(input);
        System.out.println(output2);
    }

    public static void testMatrix()
    {
        Matrix a = new Matrix(2, 2);
        Matrix b = new Matrix(2, 2);

        a.setData(new float[][] {{1 , 2}});
        b.setData(new float[][] {{-1, 0}, {1, 2}});

        System.out.println(a);
        System.out.println(b);
//        System.out.println(Matrix.add(a, b));
        System.out.println(Matrix.multiply(a, b));
//        System.out.println(Matrix.randomMatrix(2, 2));
    }
}
