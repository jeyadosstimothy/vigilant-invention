from abc import ABC, abstractmethod
from trainers import ProbenetTrainer


class Characterizer(ABC):
    @abstractmethod
    def characterize(self, dataset):
        pass


class ProbenetCharacterizer(Characterizer):
    def __init__(self):
        self.probenet = ProbenetTrainer()

    def characterize(self, dataset, epochs, batch_size):
        self.probenet.set_dataset(dataset)
        self.probenet.evaluate(epochs=epochs, batch_size=batch_size)
        return self.probenet.val_acc
