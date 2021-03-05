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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class ReinforcementLearningAgent extends Agent {

    // Store my and enemies unit ids
    private List<Integer> myUnitIDs;
    private List<Integer> enemyUnitIDs;

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

    public int[] observeState(Integer unitID, State.StateView state)
    {
        // The state consists of 2 values: the x and y components of the enemy
        // relative to the current location
        // The agent sees in a 5x5 grid around itself with the top left being 0,0
        // If the enemy is too far away in one direction that direction gets a 5 + 1 = 6
        // while the other value is 2 (reserved value, because enemy would be overlapping unit)
        // which is impossible
        int[] environment = new int[2];

        int range = 2;

        Integer enemyID = enemyUnitIDs.get(0);

        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(enemyID);

        int dx = enemy.getXPosition() - unit.getXPosition();
        int dy = enemy.getYPosition() - unit.getYPosition();

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

        environment[0] = dx;
        environment[1] = dy;

        return environment;
    }

    public float findLastReward(Integer unitID, State.StateView state, History.HistoryView history)
    {
        float stepReward = -0.001f;
        float distanceReward = 0.0009f;
        float damageReward = 0.005f;
        float enemyDamageReward = -0.001f;
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
            if (playernum == damageLog.getAttackerController()) {
                reward += damageReward * damage; // Add to reward
            }
            else  // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public float findFinalLastReward(State.StateView state, History.HistoryView history)
    {
        float reward = 0;
        float stepReward = -0.001f;
        float damageReward = 0.005f;
        float enemyDamageReward = -0.001f;
//        float enemyDamageReward = -0.00f;
        float winReward = 1.0f;
        float loseReward = 0.0f;

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
            if (playernum == damageLog.getAttackerController()) {
                reward += damageReward * damage; // Add to reward
            }
            else  // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public void requestAction(Integer unitID, int action, Map<Integer, Action> actions)
    {
        if (action >= 0 && action < 4) {
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
            Integer enemyID = enemyUnitIDs.get(0);
            actions.put(unitID, Action.createPrimitiveAttack(unitID, enemyID));
        }
        else
        {
            System.err.println("Error: Bad action " + action);
        }


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

        // And list of enemy units
        this.enemyUnitIDs = state.getUnitIds(this.enemyPlayerNum);

//        players.getCurrentPlayer().middleFitnessUpdate(state, history, myUnitIDs, enemyUnitIDs);

        // Run each unit's neural network
        // They are all the same network,
        // But each unit sees different things
        if (myUnitIDs.size() > 0 && enemyUnitIDs.size() > 0)
        {
            Integer unitID = myUnitIDs.get(0);
            int[] environment = observeState(unitID, state);
            float lastReward = findLastReward(unitID, state, history);
            int action = client.sendData(environment, lastReward);
            requestAction(unitID, action, actions);
        }

        return actions;
    }

    @Override
    public void terminalStep(State.StateView state, History.HistoryView history) {
        // Update unit IDs in case someone died and that ended the epoch
        this.myUnitIDs = state.getUnitIds(playernum);
        this.enemyUnitIDs = state.getUnitIds(enemyPlayerNum);

        float lastReward = findFinalLastReward(state, history);
        int action = client.sendData(new int[] {}, lastReward);

    }
}
