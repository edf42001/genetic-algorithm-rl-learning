package network;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

import network.layers.Layer;
import network.math.Matrix;
import network.math.MyRand;

public class Network implements Serializable {
    private ArrayList<Layer> layers;

    public Network()
    {
        this.layers = new ArrayList<Layer>(4);
    }

    public void addLayer(Layer layer)
    {
        layers.add(layer);
    }

    public Matrix feedForward(Matrix inputData)
    {
        // Propagate the input through the network

        Matrix next = inputData;
        for (Layer layer : layers)
        {
            next = layer.feedForward(next);
        }

        return next;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

//    /**
//     * Crosses over two networks
//     * @param other The other network this is being crossed with
//     * @return New crossed network
//     */
//    public Network naiveCrossover(Network other) {
//        Network child = new Network();
//
//        for (int i = 0; i < this.getLayers().size(); i++) {
//            Layer layerA = this.getLayer(i);
//            Layer layerB = other.getLayer(i);
//
//            child.addLayer(layerA.crossover(layerB));
//        }
//
//        return child;
//    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * @return Network parameters flattened into an array
     */
    public float[] getChromosome()
    {
        float[] chromosome = new float[numParams()];

        // Index to insert next layer into
        int index = 0;
        for (Layer l : layers)
        {
            l.insertIntoArray(chromosome, index);
            index += l.numParams();
        }

        return chromosome;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public Network crossover(Network other) {
        Network child = new Network();

        // Number of values in network
        int params = numParams();

        // Array representation of weights and biases
        float[] thisGenes = getChromosome();
        float[] otherGenes = other.getChromosome();
        float[] childGenes = new float[params];

        // Random crossing point
        int swapPoint = MyRand.randInt(params);

        // Do the swap
        for (int i = 0; i < swapPoint; i++)
        {
            childGenes[i] = thisGenes[i];
        }

        for(int i = swapPoint; i < params; i++)
        {
            childGenes[i] = otherGenes[i];
        }

        // Convert chromosome back into a network
        int index = 0;
        for (Layer l : this.layers) {
            Layer newLayer = l.fromLargerArray(childGenes, index);

            index += newLayer.numParams();
            child.addLayer(newLayer);
        }

        return child;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public void mutate(float mutationRate, float mutationSize)
    {
        for (Layer layer : this.layers) {
            layer.mutate(mutationRate, mutationSize);
        }

    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Save a network object to a file
     * @param file Filepath
     * @param network network.Network object to save
     */
    public static void saveNetwork(String file, Network network)
    {
        try {
            FileOutputStream fileOut = new FileOutputStream(file);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(network);
            out.close();
            fileOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Read a network object from a file
     * @param file Filepath
     * @return network.Network stored in file, or null if error
     */
    public static Network loadNetwork(String file)
    {
        try {
            FileInputStream fileIn = new FileInputStream(file);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            Network network = (Network) in.readObject();
            in.close();
            fileIn.close();
            return network;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public int numParams()
    {
        int params = 0;
        for (Layer l : layers)
        {
            params += l.numParams();
        }
        return params;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public Network clone()
    {
        Network ret = new Network();
        for (Layer layer : this.layers)
        {
            ret.addLayer(layer.clone());
        }
        return ret;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public Layer getLayer(int index)
    {
        return layers.get(index);
    }
}
