import java.io.*;
import java.util.ArrayList;

public class Network implements Serializable {
    private ArrayList<DenseLayer> layers;

    public Network()
    {
        this.layers = new ArrayList<DenseLayer>(4);
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
        return next;
    }

    /**
     * Save a network object to a file
     * @param file Filepath
     * @param network Network object to save
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
     * @return Network stored in file, or null if error
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

    public ArrayList<DenseLayer> getLayers() {
        return layers;
    }

    public DenseLayer getLayer(int index)
    {
        return layers.get(index);
    }
}
