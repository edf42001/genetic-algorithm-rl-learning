import java.util.ArrayList;

public class Network {
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
}
