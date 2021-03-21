class Agent:
    def __init__(self):
        # How many iterations this agent has been training for
        self.iterations = 0

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

    def callback(self, request):
        """Receives data from the environment, do learning and return actions here """
        raise NotImplementedError(type(self).__name__ + " callback method not implemented")

    def save_to_file(self, file):
        raise NotImplementedError(type(self).__name__ + " save_to_file method not implemented")

    def load_from_file(self, file):
        raise NotImplementedError(type(self).__name__ + " load_from_file method not implemented")

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode
