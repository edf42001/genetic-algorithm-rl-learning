import sun.nio.ch.Net;

import java.util.ArrayList;
import java.util.Random;

/**
 * Population manager
 * Runs genetic algorithm
 */
public class Population {
    // Population of individuals
    Network[] population;

    // Fitness scores for each individual
    int[] fitnesses;

    // Number of members in population
    int populationSize;

    // Which network are we testing right now and its index
    Network currentNetwork;
    int currentIndex;

    // Generation count
    int epoch;

    // Genetic evolution config
    float mutationRate = 0.005f;
    float mutationStepSize = 1.0f;
    float elitePercent = 0.1f;

    Random random = new Random();

    public Population(int populationSize)
    {
        this.population = new Network[populationSize];

        for (int i = 0; i < populationSize; i++) {
            // TODO Network features are hardcoded in the population class? That's weird
            // Create random population
            Network network = new Network();
            network.addLayer(new DenseLayer(12, 16, "relu"));
            network.addLayer(new DenseLayer(16, 16, "relu"));
            network.addLayer(new DenseLayer(16, 16, "sigmoid"));
            population[i] = network;
        }

        this.populationSize = populationSize;
        this.currentIndex = 0;
        this.currentNetwork = population[0];
        this.epoch = 0;
        this.fitnesses = new int[populationSize];
    }

    /**
     * Pull the next network out for testing
     * @return True if next epoch has occurred
     */
    public boolean moveToNextMember() {
        // If all members tested
        if (currentIndex == populationSize - 1)
        {
            selectNextPopulation();
            currentIndex = 0;
            currentNetwork = population[0];
            epoch += 1;
            return true;
        }
        // Otherwise
        currentIndex += 1;
        currentNetwork = population[currentIndex];
        return false;
    }

    /**
     * Gets what the current network wants to do
     * @param inputData Data from the environment
     * @return Output action data from network
     */
    public Matrix getActions(Matrix inputData)
    {
        return currentNetwork.feedForward(inputData);
    }

    /**
     * Genetically recombine fittest individuals
     * to form the next population
     */
    public void selectNextPopulation()
    {
        // Cumulative weights, used for roullete selection
        int[] cumulative = new int[populationSize];
        cumulative[0] = fitnesses[0];
        for (int i = 1; i < populationSize; i++) {
            cumulative[i] = fitnesses[i] + cumulative[i-1];
        }

        // Next generation
        Network[] nextPop = new Network[populationSize];

        // Create N new baby networks
        for (int i = 0; i < populationSize; i++) {
            // Choose parents based on fitness randomly
            Network parent1 = population[randomIndex(cumulative)];
            Network parent2 = population[randomIndex(cumulative)];

            // Crossover and mutate the baby
            Network baby = Population.crossover(parent1, parent2);
            mutate(baby);
            nextPop[i] = baby;  // Add to new population
        }

        population = nextPop;
    }

    public int randomIndex(int[] cumulative)
    {
        int randValue = random.nextInt(cumulative.length);
        for (int i = 0; i < cumulative.length; i++) {
            if (randValue < cumulative[i]) {
                return i;
            }
        }
        // Should never reach here
        return cumulative.length - 1;
    }

    public static Network crossover(Network a, Network b)
    {
        // TODO crossover
        return a;
    }

    public void mutate(Network network)
    {
        // TODO crossover
        for (DenseLayer layer : network.getLayers()) {
            float[][] wData = layer.getWeights().getData();
            float[][] bData = layer.getBiases().getData();

            // TODO more effecient mutaiton method?
            // Maybe choose number of mutations

            // Mutate weights
            for (int r = 0; r < wData.length; r++) {
                for (int c = 0; c < wData[0].length; c++) {
                    if (random.nextFloat() < mutationRate) {
                        wData[r][c] += mutationStepSize * 2 * (random.nextFloat() - 0.5);
                    }
                }
            }

            // Mutate biases
            for (int c = 0; c < bData[0].length; c++) {
                if (random.nextFloat() < mutationRate) {
                    bData[0][c] += mutationStepSize * 2 * (random.nextFloat() - 0.5);
                }
            }
        }
    }

    public int getEpoch() {
        return epoch;
    }

    public Network getMember(int index)
    {
        return population[index];
    }
}
