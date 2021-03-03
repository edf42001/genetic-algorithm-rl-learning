"""
The Python implementation of the GRPC RL Learning Data reciever server
recieves environment and reward data from the java SEPIA environment
and sends back an action
"""

from concurrent import futures
import logging

import grpc

import rl_environment_data_pb2
import rl_environment_data_pb2_grpc


class EnvironmentServiceImpl(rl_environment_data_pb2_grpc.EnvironmentService):

    def SendEnvironment(self, request, context):
        print("Recieved data:")
        print(request)
        return rl_environment_data_pb2.ActionResponse(action=3)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    rl_environment_data_pb2_grpc.add_EnvironmentServiceServicer_to_server(EnvironmentServiceImpl(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt as e:
        print("Keyboard interrupt, shutting down server")


if __name__ == '__main__':
    logging.basicConfig()
    serve()
