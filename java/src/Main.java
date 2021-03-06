import genetics.Population;
import network.math.Matrix;
import network.Network;
import network.layers.DenseLayer;
import network.math.MyRand;

public class Main {

    public static void main(String[] args) {
//        Population pop = new Population(1);
//        Network member = pop.getMember(0);
//
//        System.out.println(member.getLayer(0).getBiases());
//        System.out.println(member.getLayer(1).getBiases());
//        System.out.println(member.getLayer(2).getBiases());
//
//        pop.mutate(member);
//        System.out.println("After mutation");
//        System.out.println(member.getLayer(0).getBiases());
//        System.out.println(member.getLayer(1).getBiases());
//        System.out.println(member.getLayer(2).getBiases());

        testMatrix();
    }

    public static void testSavingToFIle()
    {
        DenseLayer layer0 = new DenseLayer(2, 3, "none");
        DenseLayer layer1 = new DenseLayer(3, 2, "relu");
        Network network = new Network();
        network.addLayer(layer0);
        network.addLayer(layer1);

        Network.saveNetwork("saved_data/network.ser", network);

        Network network2 = Network.loadNetwork("saved_data/network.ser");

        System.out.println(network2.getLayer(0).getWeights());
    }

    public static void testNetwork()
    {
        DenseLayer layer = new DenseLayer(3, 2, "none");
        DenseLayer layer2 = new DenseLayer(2, 1, "none");

        Network network = new Network();
        network.addLayer(layer);
        network.addLayer(layer2);

        System.out.println(layer.getWeights());
        System.out.println(layer.getBiases());

        System.out.println(layer2.getWeights());
        System.out.println(layer2.getBiases());

        Matrix input = new Matrix(new float[][] {{1, 2, 3}});

        Matrix output = layer.feedForward(input);
        System.out.println(output);

        Matrix output2 = network.feedForward(input);
        System.out.println(output2);
    }

    public static void testMatrix()
    {
        Matrix a = new Matrix(new float[][] {{1 , 2}, {1, 2}});
        Matrix b = new Matrix(new float[][] {{-1, 0}, {1, 2}});

        System.out.println(a);
        System.out.println(b);
        System.out.println(a.dot(b));

        System.out.println(a.dot(b).add(3));

        System.out.println(a.multiply(0.5f));
        System.out.println(b.multiply(0.5f));

        System.out.println(a.multiply(0.5f).add(b.multiply(0.5f)));


//        MyRand.initialize();
//        System.out.println(Msatrix.randomUniform(2, 2, 0.5f));
    }
}
