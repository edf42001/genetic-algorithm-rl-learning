package network;

import java.io.*;
import java.util.ArrayList;

import network.layers.DenseLayer;
import network.layers.Layer;
import network.math.Matrix;

public class Network implements Serializable {
    private ArrayList<DenseLayer> layers;

    // Store the last N outputs of network as memory
    // To be fed back in
    int nMemory;
    float[] memory;

    public Network(int nMemory)
    {
        this.layers = new ArrayList<DenseLayer>(4);
        this.nMemory = nMemory;
        memory = new float[nMemory];
    }

    public Network()
    {
        this(0);
    }

    public void addLayer(DenseLayer layer)
    {
        layers.add(layer);
    }

    public Matrix feedForward(Matrix inputData)
    {
        // Propagate the input through the network

        Matrix next = inputData;
        for (DenseLayer layer : layers)
        {
            next = layer.feedForward(next);
        }

        // Store N memories
        for (int i = 0; i < nMemory; i++)
        {
            memory[i] = next.getData()[0][next.getData()[0].length - 1 - i];
        }

        return next;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Crosses over two networks
     * @param other The other network this is being crossed with
     * @return New crossed network
     */
    public Network crossover(Network other) {
        Network child = new Network(nMemory);

        for (int i = 0; i < this.getLayers().size(); i++) {
            DenseLayer layerA = this.getLayer(i);
            DenseLayer layerB = other.getLayer(i);
            Matrix aWeights = layerA.getWeights();
            Matrix aBiases = layerA.getBiases();
            Matrix bWeights = layerB.getWeights();
            Matrix bBiases = layerB.getBiases();

            Matrix newWeights = aWeights.crossover(bWeights);
            Matrix newBiases = aBiases.crossover(bBiases);

            DenseLayer childLayer = new DenseLayer(newWeights, newBiases, layerA.getActivation());
            child.addLayer(childLayer);
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

    public Network clone()
    {
        Network ret = new Network(nMemory);
        for (DenseLayer layer : this.layers)
        {
            ret.addLayer(layer.clone());
        }
        return ret;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public ArrayList<DenseLayer> getLayers() {
        return layers;
    }

    public DenseLayer getLayer(int index)
    {
        return layers.get(index);
    }

    public float[] getMemory() {
        return memory;
    }
}
