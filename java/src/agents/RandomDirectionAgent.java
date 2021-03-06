package agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.util.Direction;
import network.math.MyRand;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RandomDirectionAgent extends Agent {

    // Store my and enemies unit ids
    private List<Integer> myUnitIDs;

    private int randomDirection;

    public RandomDirectionAgent(int player, String[] args) {
        super(player);

        MyRand.initialize();
    }

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

        if (state.getTurnNumber() % 3 == 0) {
            randomDirection = MyRand.randInt(4);
        }

        Direction dir = Direction.EAST; // Default value so it compiles
        switch (randomDirection) {
            case 0:
                dir = Direction.NORTH;
                break;
            case 1:
                dir = Direction.SOUTH;
                break;
            case 2:
                dir = Direction.EAST;

                break;
            case 3:
                dir = Direction.WEST;
                break;
            default:
                System.err.println("Error: Bad movement index " + randomDirection);
                break;
        }

        for (Integer unitID : myUnitIDs)
        {
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        }


        return actions;
    }

    @Override
    public void terminalStep(State.StateView state, History.HistoryView history) {

    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }


}
