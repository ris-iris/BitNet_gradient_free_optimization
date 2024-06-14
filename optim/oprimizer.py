class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, model, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def step(self, input_ids, labels, track_ops=False):
        pass

    def op_per_step(self, batch_size, seq_length):
        pass