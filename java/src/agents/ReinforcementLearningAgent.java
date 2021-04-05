package agents;

import agents.interfaces.AgentInterface;
import agents.interfaces.CrossEntropyAgentInterface;
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
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

    // Used to check actions of units who died last turn
    private List<Integer> lastMyUnitIDs;
    private List<Integer> lastEnemyUnitIDs;

    // ID of enemy player
    private int enemyPlayerNum = 1;

    // Our grpc client
    private final EnvironmentServiceClient client;

    private final boolean useNNState = true;

    private AgentInterface agentInterface;

    public ReinforcementLearningAgent(int player, String[] args) {
        super(player);

        // Initialize random number generator with no seed
        MyRand.initialize();

        // Initialize the agent interface to interact with the game world
        agentInterface = new CrossEntropyAgentInterface(playernum);

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
            int [] unitObservation = agentInterface.observeUnitState(unitID, state, myUnitIDs, enemyUnitIDs);
            float lastReward = agentInterface.getUnitLastReward(unitID, state, history, myUnitIDs, enemyUnitIDs, false);
            client.addEnvironmentState(unitObservation, lastReward, unitID);
        }

        // Send the data to the server
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
                agentInterface.requestUnitAction(myUnitIDs.get(i), actionsResponse.get(i), actions, state, enemyUnitIDs);
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
            float lastReward = agentInterface.getUnitLastReward(unitID, state, history, myUnitIDs, enemyUnitIDs, true);
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
    }
}
