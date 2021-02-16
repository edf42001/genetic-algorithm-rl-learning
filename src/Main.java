import sun.nio.ch.Net;

import java.io.*;
import java.util.Random;

public class Main {

    public static void main(String[] args) {
        Population pop = new Population(1);
        Network member = pop.getMember(0);

        System.out.println(member.getLayer(0).getBiases());
        System.out.println(member.getLayer(1).getBiases());
        System.out.println(member.getLayer(2).getBiases());

        pop.mutate(member);
        System.out.println("After mutation");
        System.out.println(member.getLayer(0).getBiases());
        System.out.println(member.getLayer(1).getBiases());
        System.out.println(member.getLayer(2).getBiases());
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
