package agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import network.math.Matrix;
import network.math.MyRand;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ReinforcementLearningAgent extends Agent {

    // Store my and enemies unit ids
    private List<Integer> myUnitIDs;
    private List<Integer> enemyUnitIDs;

    // ID of enemy player
    private int enemyPlayerNum = 1;

    public ReinforcementLearningAgent(int player, String[] args) {
        super(player);

        // Initialize random number generator with no seed
        MyRand.initialize();

        // Read enemyPlayerNum from args
        if(args.length > 0)
        {
            this.enemyPlayerNum = new Integer(args[0]);
        }

        System.out.println("In constructor of ReinforcementLearningAgent");
    }


    @Override
    public void savePlayerData(OutputStream outputStream) {}

    @Override
    public void loadPlayerData(InputStream inputStream) {}

    @Override
    public Map<Integer, Action> initialStep(State.StateView state, History.HistoryView history) {
        return null;
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView state, History.HistoryView history) {
        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        // Get ids of my units
        this.myUnitIDs = state.getUnitIds(this.playernum);

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

//        players.getCurrentPlayer().middleFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Run each unit's neural network
        // They are all the same network,
        // But each unit sees different things
        for (Integer unitID : myUnitIDs)
        {
//            int[] environment = observeState(unitID);

//            Matrix inputData = players.getCurrentPlayer().observeEnvironment(state, unitID, myUnitIDs, enemyUnitIDs);
//            players.getCurrentPlayer().standardizeInputData(inputData);
//            Matrix output = players.getActions(inputData);
//            players.getCurrentPlayer().convertOutputToActions(state, output.getData()[0], actions, unitID, enemyUnitIDs);
        }


        return actions;
    }

    @Override
    public void terminalStep(State.StateView state, History.HistoryView history) {
        // Update unit IDs in case someone died and that ended the epoch
        myUnitIDs = state.getUnitIds(playernum);
        enemyUnitIDs = state.getUnitIds(enemyPlayerNum);

    }
}
