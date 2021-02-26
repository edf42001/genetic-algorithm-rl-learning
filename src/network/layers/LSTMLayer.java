package network.layers;

import network.math.Activations;
import network.math.Matrix;

public class LSTMLayer extends Layer {
    private int inputSize;
    private int outputSize;

    private Matrix cellState;
    private Matrix output;

    private Matrix forgetWeights;
    private Matrix forgetBiases;

    private Matrix updateWeights;
    private Matrix updateBiases;

    private Matrix candidateWeights;
    private Matrix candidateBiases;

    private Matrix outputWeights;
    private Matrix outputBiases;


    public LSTMLayer(int inputSize, int outputSize)
    {
        // The cell state and input size are the same
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        int combinedSize = inputSize + outputSize;

        this.forgetWeights = randomWeights(combinedSize, outputSize);
        this.updateWeights = randomWeights(combinedSize, outputSize);
        this.candidateWeights = randomWeights(combinedSize, outputSize);
        this.outputWeights = randomWeights(combinedSize, outputSize);

        this.forgetBiases = randomBiases(outputSize);
        this.updateBiases = randomBiases(outputSize);
        this.candidateBiases = randomBiases(outputSize);
        this.outputBiases = randomBiases(outputSize);

        //Initialize to all 0s
        this.cellState = new Matrix(1, this.outputSize);
        this.output = new Matrix(1, this.outputSize);
    }

    public LSTMLayer(Matrix forgetWeights, Matrix forgetBiases, Matrix updateWeights, Matrix updateBiases,
                     Matrix candidateWeights, Matrix candidateBiases, Matrix outputWeights, Matrix outputBiases) {
        this.forgetWeights = forgetWeights;
        this.forgetBiases = forgetBiases;
        this.updateWeights = updateWeights;
        this.updateBiases = updateBiases;
        this.candidateWeights = candidateWeights;
        this.candidateBiases = candidateBiases;
        this.outputWeights = outputWeights;
        this.outputBiases = outputBiases;

        this.inputSize = forgetWeights.getRows();
        this.outputSize = forgetWeights.getCols();

        //Initialize to all 0s
        this.cellState = new Matrix(1, this.outputSize);
        this.output = new Matrix(1, this.outputSize);
    }

    public Matrix randomWeights(int inputSize, int outputSize)
    {
        // This range value is called the glorot_uniform initialization
        float range = (float) Math.sqrt(6.0 / (inputSize + outputSize));
        return Matrix.randomUniform(inputSize, outputSize, range);
    }

    public Matrix randomBiases(int size)
    {
        // For weights let's just use a smaller value
        float range = 0.5f;
        return Matrix.randomUniform(1, size, range);
    }

    @Override
    public Matrix feedForward(Matrix input) {
        // Combine input and last output
        Matrix combinedInOut = output.concatenateRow(input);

        // Forget gate
        Matrix forget = combinedInOut.dot(forgetWeights).add(forgetBiases);
        Activations.sigmoid(forget);

        // Input gate layer
        Matrix update = combinedInOut.dot(updateWeights).add(updateBiases);
        Activations.sigmoid(update);

        // Candidate values layer
        Matrix candidates = combinedInOut.dot(candidateWeights).add(candidateBiases);
        Activations.tanh(update);

        // Do the forget then update cell state calculation
        cellState = cellState.pointwiseMultiply(forget).add(candidates.pointwiseMultiply(update));

        Matrix outFilter = combinedInOut.dot(outputWeights).add(outputBiases);
        Activations.sigmoid(outFilter);

        // Need to tanh cell state but Activations modifies in place
        Matrix cellStateCopy = cellState.clone();
        Activations.tanh(cellStateCopy);

        this.output = outFilter.pointwiseMultiply(cellStateCopy);

        return this.output;
    }

    @Override
    public void mutate(float mutationRate, float mutationSize) {
        forgetWeights.mutate(mutationRate, mutationSize);
        forgetBiases.mutate(mutationRate, mutationSize);
        updateWeights.mutate(mutationRate, mutationSize);
        updateBiases.mutate(mutationRate, mutationSize);
        candidateWeights.mutate(mutationRate, mutationSize);
        candidateBiases.mutate(mutationRate, mutationSize);
        outputWeights.mutate(mutationRate, mutationSize);
        outputBiases.mutate(mutationRate, mutationSize);
    }

    @Override
    public Layer crossover(Layer other) {
        LSTMLayer layer = (LSTMLayer) other;
        Matrix newForgetWeights = forgetWeights.crossover(layer.forgetWeights);
        Matrix newForgetBiases = forgetBiases.crossover(layer.forgetBiases);
        Matrix newUpdateWeights = updateWeights.crossover(layer.updateWeights);
        Matrix newUpdateBiases = updateBiases.crossover(layer.updateBiases);
        Matrix newCandidateWeights = candidateWeights.crossover(layer.candidateWeights);
        Matrix newCandidateBiases = candidateBiases.crossover(layer.candidateBiases);
        Matrix newOutputWeights = outputWeights.crossover(layer.outputWeights);
        Matrix newOutputBiases = outputBiases.crossover(layer.outputBiases);

        return new LSTMLayer(newForgetWeights, newForgetBiases, newUpdateWeights, newUpdateBiases,
                             newCandidateWeights, newCandidateBiases, newOutputWeights, newOutputBiases);
    }

    @Override
    public Layer clone() {
        return new LSTMLayer(forgetWeights.clone(), forgetBiases.clone(), updateWeights.clone(), updateBiases.clone(),
                             candidateWeights.clone(), candidateBiases.clone(), outputWeights.clone(), outputBiases.clone());
    }

    @Override
    public Matrix getWeights() {
        return outputWeights;
    }

    @Override
    public Matrix getBiases() {
        return outputBiases;
    }
}
