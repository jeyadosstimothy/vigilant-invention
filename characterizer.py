from abc import ABC, abstractmethod
from trainers import ProbenetTrainer


class Characterizer(ABC):
    @abstractmethod
    def characterize(self, dataset):
        pass


class ProbenetCharacterizer(Characterizer):
    def __init__(self):
        self.probenet = ProbenetTrainer()

    def characterize(self, dataset):
        self.probenet.set_dataset(dataset)
        self.probenet.evaluate(epochs=3, batch_size=128)
        return self.probenet.val_acc


