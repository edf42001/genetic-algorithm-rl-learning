"""
The Python implementation of the GRPC RL Learning Data reciever server
recieves environment and reward data from the java SEPIA environment
and sends back an action
"""
import traceback
from signal import signal, SIGTERM
from concurrent import futures

import grpc
import logging
from os import kill, getpid

from protos.rl_environment_data_pb2 import ActionResponse, Empty
from protos.rl_environment_data_pb2_grpc import EnvironmentService, add_EnvironmentServiceServicer_to_server

from data_saving.data_saver import DataSaver


class EnvironmentServiceImpl(EnvironmentService):

    def __init__(self, env_callback, winner_callback):
        self.env_callback = env_callback
        self.winner_callback = winner_callback

        self.server = None

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
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
        add_EnvironmentServiceServicer_to_server(self, self.server)
        self.server.add_insecure_port('[::]:50051')
        self.server.start()

        # Sigterm handler
        def handle_sigterm(*_):
            # Shut down. Wait 3 seconds for request to finish processing
            print("Server received SIGTERM shutdown signal")
            all_rpcs_done_event = self.server.stop(3)
            all_rpcs_done_event.wait(3)
            print("Server shut down gracefully")

        signal(SIGTERM, handle_sigterm)

        try:
            self.server.wait_for_termination()
        except KeyboardInterrupt as e:
            print("Server received keyboard interrupt, shutting down")

    def stop(self):
        # Send a signal to ourselves to gracefully shutdown the server
        kill(getpid(), SIGTERM)
