import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.UnitTemplate.UnitTemplateView;
import edu.cwru.sepia.util.Direction;

import java.io.*;
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
    FileWriter myWriter;

    Network network;

    public MyCombatAgent(int player, String[] args) {
        super(player);

        // Read enemyPlayerNum from args
        if(args.length > 0)
        {
            this.enemyPlayerNum = new Integer(args[0]);
        }

        System.out.println("In constructor of MyCombatAgent");
        network = new Network();
        network.addLayer(new DenseLayer(12, 16, "relu"));
        network.addLayer(new DenseLayer(16, 16, "relu"));
        network.addLayer(new DenseLayer(16, 16, "sigmoid"));

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
        System.out.println("In agent initialStep");

        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();
//
//        // Get ids of my units
//        List<Integer> myUnitIDs = state.getUnitIds(this.playernum);
//
//        // And list of enemy units
//        List<Integer> enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);
//
//        // Nothing to do, no enemies left
//        if(enemyUnitIDs.size() == 0)
//        {
//            return actions;
//        }
//
//        // Initialize all agents to attack the 0th enemy unit
//        for(Integer myUnitID : myUnitIDs)
//        {
//            actions.put(myUnitID, Action.createCompoundAttack(myUnitID, enemyUnitIDs.get(0)));
//
//            // Get names to print message
//            UnitView unit = state.getUnit(myUnitID);
//            UnitTemplateView unitTemplate = unit.getTemplateView();
//            String unitName = unitTemplate.getName();
//
//            UnitView enemyUnit = state.getUnit(enemyUnitIDs.get(0));
//            UnitTemplateView enemyUnitTemplate = enemyUnit.getTemplateView();
//            String enemyUnitName = enemyUnitTemplate.getName();
//
//            System.out.println("Sending " + unitName + " to attack " + enemyUnitName);
//        }
//
        return actions;
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView state, History.HistoryView history) {
        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        // Get ids of my units
        this.myUnitIDs = state.getUnitIds(this.playernum);

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

        // Nothing to do, no enemies left
        if(enemyUnitIDs.size() == 0)
        {
            return actions;
        }

        Matrix inputData = observeEnvironment(state);
        System.out.print("Input data: " + inputData);

        Matrix results = network.feedForward(inputData);
        System.out.println("Network result: " + results);

        // Save input data to file
        try {
            myWriter.append(inputData.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }

        int turnNum = state.getTurnNumber();

        // Add actions to do based on network results
        convertOutputToActions(results.getData()[0], actions);
//        // Get history of current action execution
//        for(ActionResult feedback : history.getCommandFeedback(this.playernum, turnNum - 1).values())
//        {
//            // If the action was interrupted or the unit
//            // it was attacking dies, then need a new action
//            if(feedback.getFeedback() != ActionFeedback.INCOMPLETE)
//            {
//                // Get our unit id, have it switch to first enemy in list
//                int unitID = feedback.getAction().getUnitId();
//
//                // Get names to print message
//                UnitView unit = state.getUnit(unitID);
//                UnitTemplateView unitTemplate = unit.getTemplateView();
//                String unitName = unitTemplate.getName();
//
//                UnitView enemyUnit = state.getUnit(enemyUnitIDs.get(0));
//                UnitTemplateView enemyUnitTemplate = enemyUnit.getTemplateView();
//                String enemyUnitName = enemyUnitTemplate.getName();
//
//                System.out.println(unitName + unitID + " action finished with " + feedback.getFeedback());
//                System.out.println(unitName + unitID + " to attack " + enemyUnitName + enemyUnitIDs.get(0));
//            }
//        }

        return actions;
    }

    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
        System.out.println("In agent terminal step");

        // Close file
        try {
            myWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }

    float[][] standardizeInputData(float[][] inputData)
    {
        // Hardcoded mean and std dev values
        // Choose so inputs are in rance (-1, 1)
        float[] means = {12, 10, 25};
        float[] stds = {12, 10, 25};
        int nDataPerUnit = 3;

        float[][] out = new float[inputData.length][inputData[0].length];
        for (int i = 0; i< inputData[0].length; i++)
        {
            int index = i % nDataPerUnit;
            // Standardize: todo: use multiplication not division for speed
            out[0][i] = (inputData[0][i] - means[index]) / stds[index];
        }

        return out;
    }

    /**
     * Interprets the outputs of the neural network to different
     * actions and puts them in the actions object.
     * @param data output of network, in 1D array form
     * @param actions Resulting actions get put here
     */
    void convertOutputToActions(float[] data, Map<Integer, Action> actions)
    {
        // each unit takes 8 action slots:
        // move NE, SE, SW, NW, Attack, 3 slots for which unit to attack
        int nActionsPerUnit = 8;

        for (int i = 0; i < this.myUnitIDs.size(); i++)
        {
            Integer unitID = this.myUnitIDs.get(i);
            // First four determine movement direction.
            // Take max, or no movement if no >0.5
            int maxIndex = -1;
            float maxValue = 0;
            for (int j = nActionsPerUnit * i; j < nActionsPerUnit * i + 4; j++) {
                if (data[j] > 0.5 && data[j] > maxValue) {
                    maxIndex = j - nActionsPerUnit * i;
                    maxValue = data[j];
                }
            }

            if (maxIndex != -1) {
                // Select a direction from the index
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

            // Unit wants to attack if the output is on
            if (data[nActionsPerUnit * i + 5] > 0.5) {
                // Find max value, this indicates which enemy to attack
                maxIndex = -1;
                maxValue = -1;
                for (int j = nActionsPerUnit * i + 5; j < nActionsPerUnit * i + 5 + enemyUnitIDs.size(); j++) {
                    if (data[j] > maxValue) {
                        maxIndex = j - nActionsPerUnit * i - 5;
                        maxValue = data[j];
                    }
                }

                Integer enemyID = this.enemyUnitIDs.get(maxIndex);
                actions.put(unitID, Action.createCompoundAttack(unitID, enemyID));
            }
        }
    }

    /**
     * Reads data from the environment
     * @return Matrix of environment observations to be fed to network
     */
    public Matrix observeEnvironment(State.StateView state)
    {

        // Input data to neural network
        float[][] data = new float[1][12];

        for (int i = 0; i< this.myUnitIDs.size(); i++)
        {
            // Collect data on unit
            UnitView unit = state.getUnit(this.myUnitIDs.get(i));
            UnitTemplateView unitTemplate = unit.getTemplateView();
            String unitName = unitTemplate.getName();
            int unitHealth = unit.getHP();
            int unitX = unit.getXPosition();
            int unitY = unit.getYPosition();
            int unitRange = unitTemplate.getRange();
            int unitDamage = unitTemplate.getBasicAttack();

            data[0][3*i] = unitX;
            data[0][3*i+1] = unitY;
            data[0][3*i+2] = unitHealth;
//            data[0][i+3] = unitRange; // dont need, constant for now
//            data[0][i+4] = unitDamage; // same TODO add canMove?


//            System.out.printf("Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
        }

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

            // Assume two of our units with 3 data points each
            data[0][3*i+6] = unitX;
            data[0][3*i+7] = unitY;
            data[0][3*i+8] = unitHealth;

//            System.out.printf("Enemy Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
        }
        // Create matrix with input data
        Matrix inputData = new Matrix(1, 12);
        inputData.setData(standardizeInputData(data));

        return inputData;
    }

}
