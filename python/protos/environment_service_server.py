"""
The Python implementation of the GRPC RL Learning Data reciever server
recieves environment and reward data from the java SEPIA environment
and sends back an action
"""
import traceback
from concurrent import futures
import logging

import grpc

import rl_environment_data_pb2
import rl_environment_data_pb2_grpc


class EnvironmentServiceImpl(rl_environment_data_pb2_grpc.EnvironmentService):

    def __init__(self, callback):
        self.callback = callback

    def SendEnvironment(self, request, context):
        action = None  # None indicates error
        try:
            action = self.callback(request)
            return rl_environment_data_pb2.ActionResponse(action=action)
        except Exception as e:
            traceback.print_exc()
            # Not returning anything will indicate to java
            # that something went wrong

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        rl_environment_data_pb2_grpc.add_EnvironmentServiceServicer_to_server(self, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        try:
            server.wait_for_termination()
        except KeyboardInterrupt as e:
            print("Keyboard interrupt, shutting down server")


def callback(request):
    # print("Recieved data:")
    # print(request)
    return 1





if __name__ == '__main__':
    logging.basicConfig()
    serve()
