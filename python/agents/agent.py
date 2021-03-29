class Agent:
    """
    The base Agent class. Has an env_callback that receives the current state and last reward, and returns an action,
    and an end of episode winner_callback that indicates the episode has ended.
    """
    def __init__(self):
        # How many iterations this agent has been training for
        self.iterations = 0

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

    def env_callback(self, request):
        """Receives data from the environment, do learning and return actions here """
        raise NotImplementedError(type(self).__name__ + " env callback method not implemented")

    def winner_callback(self, request):
        """Receives data from the environment, do learning and return actions here """
        raise NotImplementedError(type(self).__name__ + " winner callback method not implemented")

    def save_to_file(self, folder):
        raise NotImplementedError(type(self).__name__ + " save_to_file method not implemented")

    def load_from_folder(self, folder):
        raise NotImplementedError(type(self).__name__ + " load_from_folder method not implemented")

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode

    def should_save_to_folder(self):
        raise NotImplementedError(type(self).__name__ + " should_save_to_folder method not implemented")
