package protos;

import protos.RlEnvironmentData.Environment;
import protos.RlEnvironmentData.EnvironmentRequest;
import protos.RlEnvironmentData.WinnerRequest;
import protos.RlEnvironmentData.ActionResponse;

import protos.EnvironmentServiceGrpc;

import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A client that send environment state and last action's reward to a python server
 * for processing in a reinforcement learning algorithm
 */
public class EnvironmentServiceClient {
    private static final Logger logger = Logger.getLogger(EnvironmentServiceClient.class.getName());

    private final EnvironmentServiceGrpc.EnvironmentServiceBlockingStub blockingStub;

    private EnvironmentRequest.Builder messageBuilder;

    /** Construct client for accessing HelloWorld server using the existing channel. */
    public EnvironmentServiceClient(Channel channel) {
        // 'channel' here is a Channel, not a ManagedChannel, so it is not this code's responsibility to
        // shut it down.

        // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
        blockingStub = EnvironmentServiceGrpc.newBlockingStub(channel);

        // Create empty builder for data to be added to and sent
        messageBuilder = EnvironmentRequest.newBuilder();
    }

    /** Add a unit's state to the message to send */
    public void addEnvironmentState(int[] stateData, float reward, int unitID) {
        List<Integer> state = new ArrayList<Integer>(3);

        for (int data : stateData) {
            state.add(data);
        }

        Environment agentData = Environment.newBuilder().setLastActionReward(reward).
                addAllState(state).setUnitId(unitID).build();

        messageBuilder.addAgentData(agentData);
    }

    public List<Integer> sendData(int playerID) {
        // Get request and reset builder
        EnvironmentRequest request = messageBuilder.setPlayerId(playerID).build();
        messageBuilder = EnvironmentRequest.newBuilder();

        for (Environment e : request.getAgentDataList())
        {
//            logger.info("Will try to send reward of " + e.getLastActionReward() +
//                    ", state " + e.getStateList() + ", unitID " + e.getUnitId());
        }

        ActionResponse response;
        try {
            response = blockingStub.sendEnvironment(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return null;
        }

//        logger.info("Returned actions: " + response.getActionList());
        return response.getActionList();
    }

    public boolean sendWinner(int winner, int playerID) {
        // Get request and reset builder
        WinnerRequest request = WinnerRequest.newBuilder().setWinner(winner).setPlayerId(playerID).build();

//        logger.info("Will try to send winner " + request.getWinner());

        try {
            blockingStub.sendWinner(request);
            return true;
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return false;
        }
    }



    /**
     * Environment client to send data
     */
    public static void main(String[] args) throws Exception {
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
        try {
            EnvironmentServiceClient client = new EnvironmentServiceClient(channel);
            client.addEnvironmentState(new int[] {1, 2, 3}, 0.3f, 1);
            client.sendData(0);
        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
