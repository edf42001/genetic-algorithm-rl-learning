import network.Network;
import network.layers.DenseLayer;
import network.layers.Layer;
import network.layers.RecurrentLayer;

public class TestCrossover {
    public static void main(String[] args)
    {
        Network n1 = new Network();
        Network n2 = new Network();

        addLayers(n1);
        addLayers(n2);

        Network child = n1.crossover(n2);

        System.out.println("Num params: " + n1.numParams());

        System.out.println("Network 1");
        printNetwork(n1);

        System.out.println("Network 2");
        printNetwork(n2);

        System.out.println("Child network");
        printNetwork(child);

    }

    public static void addLayers(Network n)
    {
        n.addLayer(new RecurrentLayer(2, 3));
        n.addLayer(new DenseLayer(3, 2, "none"));
    }

    public static void printNetwork(Network n)
    {
        for (Layer l : n.getLayers())
        {
            System.out.println(l.getWeights());
            System.out.println(l.getBiases());
        }
    }
}
