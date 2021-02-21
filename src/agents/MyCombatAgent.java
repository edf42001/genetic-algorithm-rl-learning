package agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.UnitTemplate.UnitTemplateView;
import edu.cwru.sepia.util.Direction;
import genetics.Population;
import network.math.Matrix;
import network.math.MyRand;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MyCombatAgent extends Agent {

    // Store my and enemies unit ids
    private List<Integer> myUnitIDs;
    private List<Integer> enemyUnitIDs;

    // ID of enemy player
    private int enemyPlayerNum = 1;

    // Save data to files
    private FileWriter myWriter;

    private Population population;

    private final boolean watchReplay = false;

    public MyCombatAgent(int player, String[] args) {
        super(player);

        // Initialize random number generator with no seed
        MyRand.initialize();

        // Read enemyPlayerNum from args
        if(args.length > 0)
        {
            this.enemyPlayerNum = new Integer(args[0]);
        }

        // Load if watching replay, else make random
        if (watchReplay) {
            this.population = Population.loadPopulation(String.format("saved_data/populations/p%d/population_%d.ser", playernum, 250));
        } else {
            this.population = new Population(80);
        }

        System.out.println("In constructor of agents.MyCombatAgent");

        // Create files to save data
        File dataFile = new File("saved_data/netInputData.csv");
        try {
            dataFile.createNewFile();
            myWriter = new FileWriter("saved_data/netInputData.csv");
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            public void run() {
                System.out.println("In shutdown hook");
                try {
                    myWriter.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }, "Shutdown-thread"));
    }

    @Override
    public Map<Integer, Action> initialStep(State.StateView state, History.HistoryView history) {
//        System.out.println("In agent initialStep");

        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }

    /**
     * Converts input data values to the range [-1, 1]
     * Using hardcoded ranges
     * @param inputData The unstandardized data. Is modified in place
     */
    public void standardizeInputData(Matrix inputData)
    {
        // Hardcoded mean and std dev values
        // Choose so inputs are in rance (-1, 1)
        // These are: delX, delY, health for each unit
        float[] means = {0, 0, 25};
        float[] stds = {10, 10, 25};
        int nDataPerUnit = means.length;

        // Get reference to data
        float[] data = inputData.getData()[0];

        // Don't standardize three memory ones
        for (int i = 0; i< data.length - 3; i++)
        {
            // Index determines what type of data we are normalizing
            // We start with health for our unit, so it needs to be offset
            int index = (i + 2) % nDataPerUnit;
            // Standardize: todo: use multiplication not division for speed
            data[i] = (data[i] - means[index]) / stds[index];
        }
    }

    /**
     * Interprets the outputs of the neural network to different
     * actions and puts them in the actions object.
     * @param data output of network, in 1D array form
     * @param actions Resulting actions get put here
     */
    void convertOutputToActions(float[] data, Map<Integer, Action> actions, Integer unitID)
    {
        // each network provides 8 output action slots:
        // move NE, SE, SW, NW, Attack, 3 slots for which unit to attack

        // First 5 determine movement direction, or attack
        // Take max, or no action if none >0.5
        int maxIndex = -1;
        float maxValue = 0;
        for (int j = 0; j < 5; j++) {
            if (data[j] > 0.5 && data[j] > maxValue) {
                maxIndex = j;
                maxValue = data[j];
            }
        }

        // If the chosen index is one of the first 4,
        // Select a direction from the index
        if (maxIndex >=0 && maxIndex < 4) {
            Direction dir = Direction.EAST; // Default value so it compiles
            switch (maxIndex) {
                case 0:
                    dir = Direction.NORTHEAST;
                    break;
                case 1:
                    dir = Direction.SOUTHEAST;
                    break;
                case 2:
                    dir = Direction.SOUTHWEST;
                    break;
                case 3:
                    dir = Direction.NORTHWEST;
                    break;
                default:
                    System.err.println("Error: Bad movement index " + maxIndex);
                    break;
            }
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        }

        // This mean unit wants to attack
        if (maxIndex == 4)
        {
            // Find max value of indices 5-7,
            // this indicates which enemy to attack
            maxIndex = -1;
            maxValue = -1;
            for (int j = 5; j < 5 + enemyUnitIDs.size(); j++) {
                if (data[j] > maxValue) {
                    maxIndex = j - 5;
                    maxValue = data[j];
                }
            }

            // Look up the enemy id the network wanted to attack
            Integer enemyID = this.enemyUnitIDs.get(maxIndex);
            actions.put(unitID, Action.createPrimitiveAttack(unitID, enemyID));
        }
    }

    /**
     * Reads data from the environment
     * @return network.math.Matrix of environment observations to be fed to network
     */
    public Matrix observeEnvironment(State.StateView state, Integer unitID)
    {

        // Input data to neural network
        // Input consists of, in order:
        // This units's health. Then for each friendly unit:
        // Their relative x, y and health.
        // Then same for each enemy unit
        // Then 3 memory inputs from last time step
        // 1 + 3 * 1 + 3 * 2 + 3 = 13
        float[][] data = new float[1][13];

        // Where this unit is. network.Network sees other units relative to itself
        UnitView thisUnit = state.getUnit(unitID);
        int myHealth = thisUnit.getHP();
        int myX = thisUnit.getXPosition();
        int myY = thisUnit.getYPosition();

        // First input
        data[0][0] = myHealth;

        // Loop through remainder of friendly units
        int iUnit = 0; // Used to count units, as our doesn't count
        for (Integer id : this.myUnitIDs)
        {
            // Our unit isn't observed by itself
            if (!id.equals(unitID)) {
                // Collect data on unit
                UnitView unit = state.getUnit(id);
                UnitTemplateView unitTemplate = unit.getTemplateView();
                String unitName = unitTemplate.getName();
                int unitHealth = unit.getHP();
                int unitX = unit.getXPosition();
                int unitY = unit.getYPosition();
                int unitRange = unitTemplate.getRange();
                int unitDamage = unitTemplate.getBasicAttack();

                // Fill in inputs in right place
                data[0][3 * iUnit + 1] = unitX - myX;
                data[0][3 * iUnit + 2] = unitY - myY;
                data[0][3 * iUnit + 3] = unitHealth;

                iUnit++; // Next unit
//            data[0][i+3] = unitRange; // dont need, constant for now
//            data[0][i+4] = unitDamage; // same TODO add canMove?

//            System.out.printf("Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
            }
        }

        // Fill in all enemy input data
        for (int i = 0; i< this.enemyUnitIDs.size(); i++)
        {
            // Collect data on unit
            UnitView unit = state.getUnit(this.enemyUnitIDs.get(i));
            UnitTemplateView unitTemplate = unit.getTemplateView();
            String unitName = unitTemplate.getName();
            int unitHealth = unit.getHP();
            int unitX = unit.getXPosition();
            int unitY = unit.getYPosition();
            int unitRange = unitTemplate.getRange();
            int unitDamage = unitTemplate.getBasicAttack();

            // Our health + 3 data points per 1 friendly unit
            // means that enemy unit info starts at i=4
            data[0][3*i+4] = unitX - myX;
            data[0][3*i+5] = unitY - myY;
            data[0][3*i+6] = unitHealth;

//            System.out.printf("Enemy Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
        }

        float[] memory = population.getCurrentNetwork().getMemory();

        for (int i = 0; i < memory.length; i++)
        {
            data[0][10 + i] = memory[i];
        }
        // Create matrix with input data
        Matrix inputData = new Matrix(data);

        return inputData;
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView state, History.HistoryView history) {
        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        // Get ids of my units
        this.myUnitIDs = state.getUnitIds(this.playernum);

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

        population.middleFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Run each unit's neural network
        // They are all the same network,
        // But each unit sees different things
        for (Integer unitID : myUnitIDs)
        {
            Matrix inputData = observeEnvironment(state, unitID);
            standardizeInputData(inputData);
            Matrix output = population.getActions(inputData);
            convertOutputToActions(output.getData()[0], actions, unitID);
//            System.out.print("Input data: " + inputData);
//            System.out.print("network.Network result: " + output);
        }

        // Save input data to file
//        try {
//            myWriter.append(inputData.toString());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        return actions;
    }

    @Override
    public void terminalStep(State.StateView state, History.HistoryView history) {
//        System.out.println("In agent terminal step");

        population.terminalFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Need to store to print because they get rest in moveToNextMember upon new epoch
        int[] fitnesses = population.getFitnesses();
        // Switch population to test next member, record if new epoch
        boolean newEpoch = population.moveToNextMember();

        if (newEpoch) {
            System.out.println("Epoch: " + population.getEpoch());
            System.out.println("Fitnesses: " + Arrays.toString(fitnesses));

            // Only save new agents if not replaying
            if (!watchReplay) {
                if (population.getEpoch() % 50 == 0)
                {
                    String file = String.format("saved_data/populations/p%d/population_%d.ser", playernum, population.getEpoch());
                    Population.savePopulation(file, population);
                }
            }

        }
        // Close file
//        try {
//            myWriter.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }


}
