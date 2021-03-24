import json

from agents import *


def load_agent(folder):
    """
    Loads an agent subclass from a agent folder
    Uses the name in the config file to determine the type
    """

    config_file = folder + "/config.txt"

    with open(config_file, 'r') as f:
        class_name = json.load(f)["name"]

    # Create instance of class from the globals that are imported into this file
    # And load that agent from the folder
    agent = globals()[class_name]()
    agent.load_from_folder(folder)

    return agent
