# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import rl_environment_data_pb2 as rl__environment__data__pb2


class EnvironmentServiceStub(object):
    """The environment data service
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendEnvironment = channel.unary_unary(
                '/EnvironmentService/SendEnvironment',
                request_serializer=rl__environment__data__pb2.EnvironmentRequest.SerializeToString,
                response_deserializer=rl__environment__data__pb2.ActionResponse.FromString,
                )


class EnvironmentServiceServicer(object):
    """The environment data service
    """

    def SendEnvironment(self, request, context):
        """Sends data, gets an action
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EnvironmentServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendEnvironment': grpc.unary_unary_rpc_method_handler(
                    servicer.SendEnvironment,
                    request_deserializer=rl__environment__data__pb2.EnvironmentRequest.FromString,
                    response_serializer=rl__environment__data__pb2.ActionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'EnvironmentService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EnvironmentService(object):
    """The environment data service
    """

    @staticmethod
    def SendEnvironment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EnvironmentService/SendEnvironment',
            rl__environment__data__pb2.EnvironmentRequest.SerializeToString,
            rl__environment__data__pb2.ActionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)