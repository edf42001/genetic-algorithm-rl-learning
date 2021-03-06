package genetics;

import network.math.Matrix;
import network.Network;
import network.layers.DenseLayer;
import network.math.MyRand;

import java.io.*;
import java.util.Arrays;

/**
 * genetics.Population manager
 * Runs genetic algorithm
 */
public class Population implements Serializable {
    // Population of individuals
    private Player[] players;

    // Number of members in population
    private int populationSize;

    // Which player are we testing right now and its index
    private Player currentPlayer;
    private int currentIndex;

    // Generation count
    private int epoch;

    // Genetic evolution config
    private float mutationRate = 0.05f;
    private float mutationStepSize = 0.5f;
    private float elitePercent = 0.1f;
    private float randomPercent = 0.05f;

    public Population(int populationSize)
    {
        this.players = new Player[populationSize];

        for (int i = 0; i < populationSize; i++) {
            this.players[i] = new Player(0);
        }

        this.populationSize = populationSize;
        this.currentIndex = 0;
        this.currentPlayer = this.players[0];
        this.epoch = 0;
    }

    /**
     * Pull the next network out for testing
     * @return True if next epoch has occurred
     */
    public boolean moveToNextMember() {
        // If all members tested
        if (currentIndex == populationSize - 1)
        {
            naturalSelection();
            currentIndex = 0;
            currentPlayer = players[0];
            epoch += 1;
            return true;
        }
        // Otherwise
        currentIndex += 1;
        currentPlayer = players[currentIndex];
        return false;
    }

    /**
     * Gets what the current network wants to do
     * @param inputData Data from the environment
     * @return Output action data from network
     */
    public Matrix getActions(Matrix inputData)
    {
        return currentPlayer.useBrain(inputData);
    }

    /**
     * Genetically recombine fittest individuals
     * to form the next population
     */
    public void naturalSelection()
    {
        // Next generation
        Player[] nextPop = new Player[populationSize];

        // Cumulative weights, used for roulette selection
        int[] roulette = new int[populationSize];
        int[] fitnesses = getFitnesses();

        // Sort top N to select them for the next generation
        // because they are the elite
        int numElite = (int) (elitePercent * populationSize);
        int[] topNIndices = sortTopN(fitnesses, numElite);

        System.out.print("Best players fitness and id: ");
        for (int i = 0; i < Math.min(numElite, 5); i++)
        {
            System.out.print(players[topNIndices[populationSize -1 - i]].getFitness() + " ");
        }
        System.out.println();
        for (int i = 0; i < Math.min(numElite, 5); i++)
        {
            System.out.print(players[topNIndices[populationSize -1 - i]].getID() + " ");
        }
        System.out.println();

        // Extract the top N most fittest individuals for the next population
        for (int i = 0; i < numElite; i++)
        {
            nextPop[i] = players[topNIndices[populationSize - 1 - i]].clone();
        }

        // Create roulette cumulative array for random selection
        roulette[0] = fitnesses[0];
        for (int i = 1; i < populationSize; i++) {
            roulette[i] = fitnesses[i] + roulette[i-1];
        }

        // Choose some amount of random parents to reproduce
        int numRandom = (int) (randomPercent * populationSize);

        for (int i = numElite; i < numElite + numRandom; i++) {
            // Choose parents completely randomly
            Player p1 = players[MyRand.randInt(populationSize)];
            Player p2 = players[MyRand.randInt(populationSize)];

            // Crossover and mutate the baby
            Player child = p1.crossover(p2);
            child.mutate(mutationRate, mutationStepSize);
            nextPop[i] = child;  // Add to new population
        }

        // Create N new baby networks
        // (minus the elite and random already selected)
        for (int i = numElite + numRandom; i < populationSize; i++) {
            // Choose parents based on fitness randomly
            Player p1 = rouletteSelection(roulette);
            Player p2 = rouletteSelection(roulette);

            // Crossover and mutate the baby
            Player child = p1.crossover(p2);
            child.mutate(mutationRate, mutationStepSize);
            nextPop[i] = child;  // Add to new population
        }

        players = nextPop;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public Player rouletteSelection(int[] cumulative)
    {
        // Max value is the total value (last value of cumulative)
        int randValue = MyRand.randInt(cumulative[cumulative.length-1]);
        for (int i = 0; i < cumulative.length; i++) {
            if (randValue < cumulative[i]) {
                return players[i];
            }
        }
        // Should never reach here
        return players[cumulative.length - 1];
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public int getEpoch() {
        return epoch;
    }

    public Player getMember(int index)
    {
        return players[index];
    }

    public Player getCurrentPlayer() {
        return currentPlayer;
    }

    public int[] getFitnesses() {
        int[] ret = new int[populationSize];

        for (int i = 0; i < populationSize; i++)
        {
            ret[i] = players[i].getFitness();
        }
        return ret;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public static void savePopulation(String file, Population population)
    {
        try {
            FileOutputStream fileOut = new FileOutputStream(file);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(population);
            out.close();
            fileOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public static Population loadPopulation(String file)
    {
        try {
            FileInputStream fileIn = new FileInputStream(file);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            Population population = (Population) in.readObject();
            in.close();
            fileIn.close();
            return population;
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Sorts an array, but only the top N values
     * This is used to select N best members from population
     * @param data Array to sort
     * @param n Number of biggest numbers to sort
     */
    public static int[] sortTopN(int[] data, int n)
    {
        // The indices of the top N members
        // based on the fitnesses in data
        int[] indices = new int[data.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        for (int i = 0; i < n; i++)
        {
            // Do selection sort
            int maxIndex = 0;
            for (int j = 0; j < data.length - i; j++)
            {
                if (data[j] > data[maxIndex]) {
                    maxIndex = j;
                }
            }

            // Swap indices with values to keep track
            int tmp = data[data.length - 1 - i];
            int tmpI = indices[data.length - 1 - i];
            data[data.length - 1 - i] = data[maxIndex];
            indices[data.length - 1 - i] = indices[maxIndex];
            data[maxIndex] = tmp;
            indices[maxIndex] = tmpI;
        }

        return indices;
    }
}
