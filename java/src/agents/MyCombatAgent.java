package agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
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

    private Population players;

    private int epochsElapsed;

    private long startTime;

    private final boolean watchReplay = true;
    private final int epochsToEvolve = 100;


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
            this.players = Population.loadPopulation(String.format("saved_data/populations/p%d/population_%d.ser", playernum, 800));
        } else {
            this.players = new Population(500);
        }

        System.out.println("In constructor of MyCombatAgent");
        System.out.println("Parameters in brain: " + players.getCurrentPlayer().getBrain().numParams());

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
                System.out.println("Executing shutdown hook");
                try {
                    myWriter.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }, "Shutdown-thread"));

        epochsElapsed = 0;
        startTime = System.currentTimeMillis();
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


    @Override
    public Map<Integer, Action> middleStep(State.StateView state, History.HistoryView history) {
        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        // Get ids of my units
        this.myUnitIDs = state.getUnitIds(this.playernum);

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

        players.getCurrentPlayer().middleFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Run each unit's neural network
        // They are all the same network,
        // But each unit sees different things
        for (Integer unitID : myUnitIDs)
        {
            Matrix inputData = players.getCurrentPlayer().observeEnvironment(state, unitID, myUnitIDs, enemyUnitIDs);
            players.getCurrentPlayer().standardizeInputData(inputData);
            Matrix output = players.getActions(inputData);
            players.getCurrentPlayer().convertOutputToActions(state, output.getData()[0], actions, unitID, enemyUnitIDs);
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
        myUnitIDs = state.getUnitIds(playernum);
        enemyUnitIDs = state.getUnitIds(enemyPlayerNum);
        players.getCurrentPlayer().terminalFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Need to store to print because they get rest in moveToNextMember upon new epoch
        int[] fitnesses = players.getFitnesses();

        if (watchReplay) {
            System.out.println("Fitness: " + players.getCurrentPlayer().getFitness());
        }

        // Switch population to test next member, record if new epoch
        boolean newEpoch = players.moveToNextMember();

        if (newEpoch) {
            System.out.println("Epoch: " + players.getEpoch());
            System.out.println("Fitnesses: " + Arrays.toString(fitnesses));

            // One epoch has passed
            epochsElapsed += 1;

            // Only save new agents if not replaying
            if (!watchReplay) {
                if (players.getEpoch() % 50 == 0)
                {
                    String file = String.format("saved_data/populations/p%d/population_%d.ser", playernum, players.getEpoch());
                    Population.savePopulation(file, players);
                }
            }

            if (epochsElapsed >= epochsToEvolve)
            {
                System.out.println("Reached epoch " + players.getEpoch() + ", stopping");
                long currentTime = System.currentTimeMillis();
                System.out.printf("Time elapsed: %d\n", (currentTime - startTime) / 1000);
                System.exit(0);
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
