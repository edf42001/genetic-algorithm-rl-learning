package agents;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.util.Direction;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import network.math.MyRand;
import protos.EnvironmentServiceClient;

import java.io.*;
import java.sql.Time;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class ReinforcementLearningAgent extends Agent {

    // Store my and enemies unit ids
    private List<Integer> myUnitIDs;
    private List<Integer> enemyUnitIDs;

    // Used to check actions of units who died last turn
    private List<Integer> lastMyUnitIDs;
    private List<Integer> lastEnemyUnitIDs;

    // ID of enemy player
    private int enemyPlayerNum = 1;

    // Our grpc client
    private final EnvironmentServiceClient client;

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

        // Set up grpc client
        // Access a service running on the local machine on port 50051
        String target = "localhost:50051";

        // Create a communication channel to the server, known as a Channel. Channels are thread-safe
        // and reusable. It is common to create channels at the beginning of your application and reuse
        // them until the application shuts down.
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target)
                // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                // needing certificates.
                .usePlaintext()
                .build();

        client = new EnvironmentServiceClient(channel);


        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            public void run() {
                System.out.println("Executing shutdown hook");
                try {
                    // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
                    // resources the channel should be shut down when it will no longer be used. If it may be used
                    // again leave it running.
                    channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, "Shutdown-thread"));
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {}

    @Override
    public void loadPlayerData(InputStream inputStream) {}

    /**
     * The relative location of another unit consists of 2 values:
     * the x and y components of the enemy
     * The agent sees in a 5x5 grid around itself with the top left being 0,0
     * If the enemy is too far away in one direction that direction gets a 5 + 1 = 6
     * while the other value is 2 (reserved value, because enemy would be overlapping unit)
     * which is impossible
     * @param unitID This unit
     * @param otherID The other unit
     * @param state Stateview
     * @return The x, y, or offscreen values
     */
    public int[] getUnitRelativeLocation(Integer unitID, Integer otherID, State.StateView state)
    {
        int range = 2; // How far away units can see

        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(otherID);

        int dx = enemy.getXPosition() - unit.getXPosition();
        int dy = enemy.getYPosition() - unit.getYPosition();

        // Units expect to be player 1, in the left side of the field
        // Flip their state if they are on the other side of the field
        // so they get data they are used to
        if (playernum == 0)
        {
            dx *= -1;
            dy *= -1;
        }

        // If x or y is greater or less than range
        // clip the value and set the other to the reserved
        // value
        if (dx > range) {
            dx = range + 1;
            dy = 0;
        } else if (dx < -range) {
            dx = -range - 1;
            dy = 0;
        } else if (dy > range) {
            dy = range + 1;
            dx = 0;
        } else if (dy < -range) {
            dy = -range - 1;
            dx = 0;
        }

        // convert to 1 - 5
        // reserved value of 0 becomes 3 (in the middle)
        // 0 means "off to the left or down" 6 means "off to the right or up"
        dx += range + 1;
        dy += range + 1;

        return new int[] {dx, dy};
    }

    public int[] observeState(Integer unitID, State.StateView state)
    {
        // Indicates a dead unit
        int deadValue = 3;

        // 3 other units, x and y for each makes 6
        // A dead unit gets is placed at the location of
        // the current unit (which is impossible otherwise)
        // (that index is (3, 3) I believe)
        int[] environment = new int[6];

        // Get location of friendly unit
        // idx keeps track of where in state array we are
        int[] unitLoc;
        int idx = 0;
        for (Integer id : myUnitIDs)
        {
            // If not the other unit
            if (!id.equals(unitID))
            {
                unitLoc = getUnitRelativeLocation(unitID, id, state);
                environment[idx] = unitLoc[0];
                environment[idx + 1] = unitLoc[1];
                idx += 2;
            }
        }

        // If our friendly unit is dead
        // Put it in the dead state
        if (idx == 0)
        {
            environment[idx] = deadValue;
            environment[idx + 1] = deadValue;
            idx += 2;
        }

        // Fill in enemies' states
        // Or dead if they are dead
        for (Integer id : enemyUnitIDs)
        {
            unitLoc = getUnitRelativeLocation(unitID, id, state);
            environment[idx] = unitLoc[0];
            environment[idx + 1] = unitLoc[1];
            idx += 2;
        }

        // If any enemy units are dead
        // fill state in with the dead state
        while (idx < 6)
        {
            environment[idx] = deadValue;
            environment[idx + 1] = deadValue;
            idx += 2;
        }

        return environment;
    }

    public float findLastReward(Integer unitID, State.StateView state, History.HistoryView history)
    {
        float stepReward = -0.01f;
        float distanceReward = 0.0009f;
        float damageReward = 0.005f;
        float enemyDamageReward = -0.004f;
//        float enemyDamageReward = -0.00f;

        float deathReward = 1.0f;

        float reward = 0;

        // Punishment for taking too long
        reward += stepReward;

        int turnNum = state.getTurnNumber();

        // Incentivize getting close to enemy
        // Reward falls off as 1 / distance to enemy
        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(enemyUnitIDs.get(0));

        int unitX = unit.getXPosition();
        int unitY = unit.getYPosition();
        int enemyX = enemy.getXPosition();
        int enemyY = enemy.getYPosition();

        reward += distanceReward / (Math.abs(enemyX - unitX) + Math.abs(enemyY - unitY));

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public float findFinalLastReward(Integer unitID, State.StateView state, History.HistoryView history)
    {
        float reward = 0;
        float stepReward = -0.01f;
        float damageReward = 0.005f;
        float enemyDamageReward = -0.004f;
//        float enemyDamageReward = -0.00f;
        float winReward = 1.0f;
        float loseReward = -0.8f;

        // Punishment for taking too long
        reward += stepReward;

        // Reward for winning
        if (enemyUnitIDs.size() == 0)
        {
            reward += winReward;
        }

        if (myUnitIDs.size() == 0)
        {
            reward += loseReward;
        }

        int turnNum = state.getTurnNumber();

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public void requestAction(Integer unitID, int action, Map<Integer, Action> actions, State.StateView state)
    {
        if (action >= 0 && action < 4) {
            // Agents expect to be player 1, on the left of the screen
            // When an agent on the right of the screen says to go right,
            // they actually mean left. This flips left/right, up/down for that agent
            if (playernum == 0)
            {
                action = (action + 2) % 4;
            }

            Direction dir = Direction.NORTH; // Default value so it compiles
            switch (action) {
                case 0:
                    dir = Direction.NORTH;
                    break;
                case 1:
                    dir = Direction.EAST;
                    break;
                case 2:
                    dir = Direction.SOUTH;
                    break;
                case 3:
                    dir = Direction.WEST;
                    break;
                default:
                    System.err.println("Error: Bad action " + action);
                    break;
            }
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        }
        else if(action == 4)
        {
            Unit.UnitView unit = state.getUnit(unitID);

            // Find the closest enemy and attack them
            // may not be in range
            Integer closestEnemyID = enemyUnitIDs.get(0);
            int closestEnemyDist = Integer.MAX_VALUE;
            for (Integer enemyID : enemyUnitIDs)
            {
                Unit.UnitView enemy = state.getUnit(enemyID);

                int dx = enemy.getXPosition() - unit.getXPosition();
                int dy = enemy.getYPosition() - unit.getYPosition();

                if (Math.max(dx, dy) < closestEnemyDist)
                {
                    closestEnemyDist = Math.max(dx, dy);
                    closestEnemyID = enemyID;
                }
            }

            actions.put(unitID, Action.createPrimitiveAttack(unitID, closestEnemyID));
        }
        else if (action == -2)
        {
            System.err.println("Error: Something went wrong, bad action " + action);
        }
        else
        {
            System.out.println("Note: noop action " + action);
        }
    }

    @Override
    public Map<Integer, Action> initialStep(State.StateView state, History.HistoryView history) {
        return middleStep(state, history);
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView state, History.HistoryView history) {
        // Actions to do
        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        // Get ids of my units
        this.myUnitIDs = state.getUnitIds(this.playernum);

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

        // Get each unit's observations and last reward
        // Add them to the message to be sent to python
        for (Integer unitID : myUnitIDs)
        {
            int[] unitObservation = observeState(unitID, state);
            float lastReward = findLastReward(unitID, state, history);
            client.addEnvironmentState(unitObservation, lastReward, unitID);
        }

        List<Integer> actionsResponse = client.sendData(1 - playernum);
        if (actionsResponse == null)
        {
            System.err.println("Error: actionResponse null. Bad actions returned");
        }
        else if (actionsResponse.size() == 0)
        {
            System.out.println("Info: Received noop empty actions");
        }
        else
        {
            // We receive one action for every env data sent
            for (int i = 0; i < actionsResponse.size(); i++)
            {
                requestAction(myUnitIDs.get(i), actionsResponse.get(i), actions, state);
            }
        }

        this.lastMyUnitIDs = myUnitIDs;
        this.lastEnemyUnitIDs = enemyUnitIDs;
        return actions;
    }

    @Override
    public void terminalStep(State.StateView state, History.HistoryView history) {
        this.myUnitIDs = state.getUnitIds(playernum);
        this.enemyUnitIDs = state.getUnitIds(enemyPlayerNum);

        // Send the final reward for units
        for (Integer unitID : lastMyUnitIDs)
        {
            float lastReward = findFinalLastReward(unitID, state, history);
            client.addEnvironmentState(new int[] {}, lastReward, unitID);
        }

        int winner = -1;
        // Win, lose, or tie
        // Here "player0" is green, but that's player id 1, so do 1 - to swap
        if (enemyUnitIDs.size() == 0)
        {
            winner = 1 - playernum;
        }
        else if (myUnitIDs.size() == 0)
        {
            winner = 1 - enemyPlayerNum;
        }

        client.sendData(1 - playernum);
        client.sendWinner(winner, 1 - playernum);

        //DO a for int i in range to send reward for all?
        // Give winning reward to all units or jus the won that made the kill?
        // Could make unit do move that didn't have anything to do with win
        // Such as that guy who was going left for that reason most likely
        // He stole his teammate's reward
    }
}
