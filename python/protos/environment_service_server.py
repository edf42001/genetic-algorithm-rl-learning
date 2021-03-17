"""
The Python implementation of the GRPC RL Learning Data reciever server
recieves environment and reward data from the java SEPIA environment
and sends back an action
"""
import traceback
from concurrent import futures

import grpc
import logging

from protos.rl_environment_data_pb2 import ActionResponse, Empty
from protos.rl_environment_data_pb2_grpc import EnvironmentService, add_EnvironmentServiceServicer_to_server

from data_saving.data_saver import DataSaver


class EnvironmentServiceImpl(EnvironmentService):

    def __init__(self, env_callback, winner_callback):
        self.env_callback = env_callback
        self.winner_callback = winner_callback

    def SendEnvironment(self, request, context):
        action = None  # None indicates error
        try:
            action = self.env_callback(request)
            return ActionResponse(action=action)
        except Exception as e:
            traceback.print_exc()
            # Not returning anything will indicate to java
            # that something went wrong

    def SendWinner(self, request, context):
        try:
            self.winner_callback(request)
            return Empty()
        except Exception as e:
            traceback.print_exc()

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        add_EnvironmentServiceServicer_to_server(self, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        try:
            server.wait_for_termination()
        except KeyboardInterrupt as e:
            print("Keyboard interrupt, shutting down server")
