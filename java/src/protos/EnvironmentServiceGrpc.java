package protos;

import static io.grpc.MethodDescriptor.generateFullMethodName;
import static io.grpc.stub.ClientCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ClientCalls.asyncClientStreamingCall;
import static io.grpc.stub.ClientCalls.asyncServerStreamingCall;
import static io.grpc.stub.ClientCalls.asyncUnaryCall;
import static io.grpc.stub.ClientCalls.blockingServerStreamingCall;
import static io.grpc.stub.ClientCalls.blockingUnaryCall;
import static io.grpc.stub.ClientCalls.futureUnaryCall;
import static io.grpc.stub.ServerCalls.asyncBidiStreamingCall;
import static io.grpc.stub.ServerCalls.asyncClientStreamingCall;
import static io.grpc.stub.ServerCalls.asyncServerStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnaryCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedStreamingCall;
import static io.grpc.stub.ServerCalls.asyncUnimplementedUnaryCall;

/**
 * <pre>
 * The environment data service
 * </pre>
 */
@javax.annotation.Generated(
    value = "by gRPC proto compiler (version 1.15.0)",
    comments = "Source: rl_environment_data.proto")
public final class EnvironmentServiceGrpc {

  private EnvironmentServiceGrpc() {}

  public static final String SERVICE_NAME = "EnvironmentService";

  // Static method descriptors that strictly reflect the proto.
  private static volatile io.grpc.MethodDescriptor<protos.RlEnvironmentData.EnvironmentRequest,
      protos.RlEnvironmentData.ActionResponse> getSendEnvironmentMethod;

  @io.grpc.stub.annotations.RpcMethod(
      fullMethodName = SERVICE_NAME + '/' + "SendEnvironment",
      requestType = protos.RlEnvironmentData.EnvironmentRequest.class,
      responseType = protos.RlEnvironmentData.ActionResponse.class,
      methodType = io.grpc.MethodDescriptor.MethodType.UNARY)
  public static io.grpc.MethodDescriptor<protos.RlEnvironmentData.EnvironmentRequest,
      protos.RlEnvironmentData.ActionResponse> getSendEnvironmentMethod() {
    io.grpc.MethodDescriptor<protos.RlEnvironmentData.EnvironmentRequest, protos.RlEnvironmentData.ActionResponse> getSendEnvironmentMethod;
    if ((getSendEnvironmentMethod = EnvironmentServiceGrpc.getSendEnvironmentMethod) == null) {
      synchronized (EnvironmentServiceGrpc.class) {
        if ((getSendEnvironmentMethod = EnvironmentServiceGrpc.getSendEnvironmentMethod) == null) {
          EnvironmentServiceGrpc.getSendEnvironmentMethod = getSendEnvironmentMethod = 
              io.grpc.MethodDescriptor.<protos.RlEnvironmentData.EnvironmentRequest, protos.RlEnvironmentData.ActionResponse>newBuilder()
              .setType(io.grpc.MethodDescriptor.MethodType.UNARY)
              .setFullMethodName(generateFullMethodName(
                  "EnvironmentService", "SendEnvironment"))
              .setSampledToLocalTracing(true)
              .setRequestMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  protos.RlEnvironmentData.EnvironmentRequest.getDefaultInstance()))
              .setResponseMarshaller(io.grpc.protobuf.ProtoUtils.marshaller(
                  protos.RlEnvironmentData.ActionResponse.getDefaultInstance()))
                  .setSchemaDescriptor(new EnvironmentServiceMethodDescriptorSupplier("SendEnvironment"))
                  .build();
          }
        }
     }
     return getSendEnvironmentMethod;
  }

  /**
   * Creates a new async stub that supports all call types for the service
   */
  public static EnvironmentServiceStub newStub(io.grpc.Channel channel) {
    return new EnvironmentServiceStub(channel);
  }

  /**
   * Creates a new blocking-style stub that supports unary and streaming output calls on the service
   */
  public static EnvironmentServiceBlockingStub newBlockingStub(
      io.grpc.Channel channel) {
    return new EnvironmentServiceBlockingStub(channel);
  }

  /**
   * Creates a new ListenableFuture-style stub that supports unary calls on the service
   */
  public static EnvironmentServiceFutureStub newFutureStub(
      io.grpc.Channel channel) {
    return new EnvironmentServiceFutureStub(channel);
  }

  /**
   * <pre>
   * The environment data service
   * </pre>
   */
  public static abstract class EnvironmentServiceImplBase implements io.grpc.BindableService {

    /**
     * <pre>
     * Sends data, gets an action
     * </pre>
     */
    public void sendEnvironment(protos.RlEnvironmentData.EnvironmentRequest request,
        io.grpc.stub.StreamObserver<protos.RlEnvironmentData.ActionResponse> responseObserver) {
      asyncUnimplementedUnaryCall(getSendEnvironmentMethod(), responseObserver);
    }

    @java.lang.Override public final io.grpc.ServerServiceDefinition bindService() {
      return io.grpc.ServerServiceDefinition.builder(getServiceDescriptor())
          .addMethod(
            getSendEnvironmentMethod(),
            asyncUnaryCall(
              new MethodHandlers<
                protos.RlEnvironmentData.EnvironmentRequest,
                protos.RlEnvironmentData.ActionResponse>(
                  this, METHODID_SEND_ENVIRONMENT)))
          .build();
    }
  }

  /**
   * <pre>
   * The environment data service
   * </pre>
   */
  public static final class EnvironmentServiceStub extends io.grpc.stub.AbstractStub<EnvironmentServiceStub> {
    private EnvironmentServiceStub(io.grpc.Channel channel) {
      super(channel);
    }

    private EnvironmentServiceStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected EnvironmentServiceStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new EnvironmentServiceStub(channel, callOptions);
    }

    /**
     * <pre>
     * Sends data, gets an action
     * </pre>
     */
    public void sendEnvironment(protos.RlEnvironmentData.EnvironmentRequest request,
        io.grpc.stub.StreamObserver<protos.RlEnvironmentData.ActionResponse> responseObserver) {
      asyncUnaryCall(
          getChannel().newCall(getSendEnvironmentMethod(), getCallOptions()), request, responseObserver);
    }
  }

  /**
   * <pre>
   * The environment data service
   * </pre>
   */
  public static final class EnvironmentServiceBlockingStub extends io.grpc.stub.AbstractStub<EnvironmentServiceBlockingStub> {
    private EnvironmentServiceBlockingStub(io.grpc.Channel channel) {
      super(channel);
    }

    private EnvironmentServiceBlockingStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected EnvironmentServiceBlockingStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new EnvironmentServiceBlockingStub(channel, callOptions);
    }

    /**
     * <pre>
     * Sends data, gets an action
     * </pre>
     */
    public protos.RlEnvironmentData.ActionResponse sendEnvironment(protos.RlEnvironmentData.EnvironmentRequest request) {
      return blockingUnaryCall(
          getChannel(), getSendEnvironmentMethod(), getCallOptions(), request);
    }
  }

  /**
   * <pre>
   * The environment data service
   * </pre>
   */
  public static final class EnvironmentServiceFutureStub extends io.grpc.stub.AbstractStub<EnvironmentServiceFutureStub> {
    private EnvironmentServiceFutureStub(io.grpc.Channel channel) {
      super(channel);
    }

    private EnvironmentServiceFutureStub(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      super(channel, callOptions);
    }

    @java.lang.Override
    protected EnvironmentServiceFutureStub build(io.grpc.Channel channel,
        io.grpc.CallOptions callOptions) {
      return new EnvironmentServiceFutureStub(channel, callOptions);
    }

    /**
     * <pre>
     * Sends data, gets an action
     * </pre>
     */
    public com.google.common.util.concurrent.ListenableFuture<protos.RlEnvironmentData.ActionResponse> sendEnvironment(
        protos.RlEnvironmentData.EnvironmentRequest request) {
      return futureUnaryCall(
          getChannel().newCall(getSendEnvironmentMethod(), getCallOptions()), request);
    }
  }

  private static final int METHODID_SEND_ENVIRONMENT = 0;

  private static final class MethodHandlers<Req, Resp> implements
      io.grpc.stub.ServerCalls.UnaryMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ServerStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.ClientStreamingMethod<Req, Resp>,
      io.grpc.stub.ServerCalls.BidiStreamingMethod<Req, Resp> {
    private final EnvironmentServiceImplBase serviceImpl;
    private final int methodId;

    MethodHandlers(EnvironmentServiceImplBase serviceImpl, int methodId) {
      this.serviceImpl = serviceImpl;
      this.methodId = methodId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public void invoke(Req request, io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        case METHODID_SEND_ENVIRONMENT:
          serviceImpl.sendEnvironment((protos.RlEnvironmentData.EnvironmentRequest) request,
              (io.grpc.stub.StreamObserver<protos.RlEnvironmentData.ActionResponse>) responseObserver);
          break;
        default:
          throw new AssertionError();
      }
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("unchecked")
    public io.grpc.stub.StreamObserver<Req> invoke(
        io.grpc.stub.StreamObserver<Resp> responseObserver) {
      switch (methodId) {
        default:
          throw new AssertionError();
      }
    }
  }

  private static abstract class EnvironmentServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoFileDescriptorSupplier, io.grpc.protobuf.ProtoServiceDescriptorSupplier {
    EnvironmentServiceBaseDescriptorSupplier() {}

    @java.lang.Override
    public com.google.protobuf.Descriptors.FileDescriptor getFileDescriptor() {
      return protos.RlEnvironmentData.getDescriptor();
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.ServiceDescriptor getServiceDescriptor() {
      return getFileDescriptor().findServiceByName("EnvironmentService");
    }
  }

  private static final class EnvironmentServiceFileDescriptorSupplier
      extends EnvironmentServiceBaseDescriptorSupplier {
    EnvironmentServiceFileDescriptorSupplier() {}
  }

  private static final class EnvironmentServiceMethodDescriptorSupplier
      extends EnvironmentServiceBaseDescriptorSupplier
      implements io.grpc.protobuf.ProtoMethodDescriptorSupplier {
    private final String methodName;

    EnvironmentServiceMethodDescriptorSupplier(String methodName) {
      this.methodName = methodName;
    }

    @java.lang.Override
    public com.google.protobuf.Descriptors.MethodDescriptor getMethodDescriptor() {
      return getServiceDescriptor().findMethodByName(methodName);
    }
  }

  private static volatile io.grpc.ServiceDescriptor serviceDescriptor;

  public static io.grpc.ServiceDescriptor getServiceDescriptor() {
    io.grpc.ServiceDescriptor result = serviceDescriptor;
    if (result == null) {
      synchronized (EnvironmentServiceGrpc.class) {
        result = serviceDescriptor;
        if (result == null) {
          serviceDescriptor = result = io.grpc.ServiceDescriptor.newBuilder(SERVICE_NAME)
              .setSchemaDescriptor(new EnvironmentServiceFileDescriptorSupplier())
              .addMethod(getSendEnvironmentMethod())
              .build();
        }
      }
    }
    return result;
  }
}
