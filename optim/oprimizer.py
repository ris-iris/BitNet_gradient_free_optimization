class Optimizer:
    def __init__(self, model, loss_fn) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def step(self, input_ids, labels):
        pass