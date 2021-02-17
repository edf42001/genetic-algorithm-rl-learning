import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

import java.io.*;
import java.util.List;
import java.util.Random;

/**
 * Population manager
 * Runs genetic algorithm
 */
public class Population implements Serializable {
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
    float mutationRate = 0.01f;
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
            fitnesses = new int[populationSize];
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
            int index1 = randomIndex(cumulative);
            int index2 = randomIndex(cumulative);
            Network parent1 = population[index1];
            Network parent2 = population[index2];

            System.out.println("Selected indices " + index1 + ", " + index2);

            // Crossover and mutate the baby
            Network baby = crossover(parent1, parent2);
            mutate(baby);
            nextPop[i] = baby;  // Add to new population
        }

        population = nextPop;
    }

    public int randomIndex(int[] cumulative)
    {
        // Max value is the total value (last value of cumulative)
        int randValue = random.nextInt(cumulative[cumulative.length-1]);
        for (int i = 0; i < cumulative.length; i++) {
            if (randValue < cumulative[i]) {
                return i;
            }
        }
        // Should never reach here
        return cumulative.length - 1;
    }

    public Network crossover(Network a, Network b)
    {
        // TODO crossover
        // TODO this is a dumb linear combo method
        Network child = new Network();

        for (int i = 0; i < a.getLayers().size(); i++) {
            DenseLayer layerA = a.getLayer(i);
            DenseLayer layerB = b.getLayer(i);
            Matrix aWeights = layerA.getWeights();
            Matrix aBiases = layerA.getBiases();
            Matrix bWeights = layerB.getWeights();
            Matrix bBiases = layerB.getBiases();

            float ratio = random.nextFloat();
            Matrix newWeights = Matrix.add(Matrix.multiply(ratio, aWeights),
                                            Matrix.multiply(1 - ratio, bWeights));
            Matrix newBiases = Matrix.add(Matrix.multiply(ratio, aBiases),
                    Matrix.multiply(1 - ratio, bBiases));

            DenseLayer childLayer = new DenseLayer(newWeights, newBiases, layerA.getActivation());
            child.addLayer(childLayer);

        }

        return child;
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

    /**
     * Uses information from the game state and last turn history
     * to find the member's fitness
     * @param state
     * @param history
     */
    public void evaluateMemberFItness(State.StateView state, History.HistoryView history,
                                      List<Integer> myUnitIDs, List<Integer> enemyUnitIDs)
    {
        // Current fitness function:
        // +1 point for every time step spent within 5 spaces of an enemy
        // This encourages agents to go near enemies (danger bonus)
        // 10 * damage dealt to opponents. This encourages agents to murder
        // And will overwhelm the danger bonus, so agents can ignore that
        // If fitness is 0 set it to 1 so the math doesn't break

        int dangerWeight = 1;
        int damageWeight = 20;
        int dangerDistance = 6;

        int turnNum = state.getTurnNumber();

        // Check danger bonus fitness
        // For every unit see if close to enemy
        for (Integer unitID : myUnitIDs)
        {
            // Get unit x and y
            Unit.UnitView unit = state.getUnit(unitID);
            int unitX = unit.getXPosition();
            int unitY = unit.getYPosition();

            for (Integer enemyUnitID : enemyUnitIDs)
            {
                Unit.UnitView enemyUnit = state.getUnit(enemyUnitID);
                int enemyUnitX = enemyUnit.getXPosition();
                int enemyUnitY = enemyUnit.getYPosition();
                if (Math.max(Math.abs(enemyUnitX - unitX), Math.abs(enemyUnitY - unitY)) < dangerDistance)
                {
                    fitnesses[currentIndex] += dangerWeight;
                    break; // Only one point per unit (not for each enemy)
                }
            }
        }

        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs)
        {
            // If we did the damage
            if (myUnitIDs.contains(damageLog.getAttackerID()))
            {
                // Add to fitness
                int damage = damageLog.getDamage();
                System.err.println("YAYAYAYAYAYYA");
                fitnesses[currentIndex] += damageWeight * damage;
            }
        }

        // Just in case
        //TODO this only needs to be done once upon creation of array
        if (fitnesses[currentIndex] == 0) {
            fitnesses[currentIndex] = 1;
        }
    }

    public int getEpoch() {
        return epoch;
    }

    public int[] getFitnesses() {
        return fitnesses;
    }

    public Network getMember(int index)
    {
        return population[index];
    }

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
}
