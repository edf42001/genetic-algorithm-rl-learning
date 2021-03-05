package protos;

import protos.RlEnvironmentData.EnvironmentData;
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

    /** Construct client for accessing HelloWorld server using the existing channel. */
    public EnvironmentServiceClient(Channel channel) {
        // 'channel' here is a Channel, not a ManagedChannel, so it is not this code's responsibility to
        // shut it down.

        // Passing Channels to code makes code easier to test and makes it easier to reuse Channels.
        blockingStub = EnvironmentServiceGrpc.newBlockingStub(channel);
    }

    /** Say hello to server. */
    public int sendData(int[] stateData, float reward) {
        List<Integer> state = new ArrayList<Integer>(3);

        for (int data : stateData) {
            state.add(data);
        }

        logger.info("Will try to send reward of " + reward + " and state " + state);
        EnvironmentData request = EnvironmentData.newBuilder().setLastActionReward(reward).addAllState(state).build();
        ActionResponse response;
        try {
            response = blockingStub.sendEnvironment(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return -1;
        }
        logger.info("Returned action: " + response.getAction());
        return response.getAction();
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
            client.sendData(new int[] {1, 2, 3}, 0.3f);
        } finally {
            // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
            // resources the channel should be shut down when it will no longer be used. If it may be used
            // again leave it running.
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
