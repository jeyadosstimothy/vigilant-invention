from abc import ABC, abstractmethod
from trainers import ProbenetTrainer


class Characterizer(ABC):
    @abstractmethod
    def characterize(self, dataset):
        pass


class ProbenetCharacterizer(Characterizer):
    def __init__(self, useTpu=False):
        self.probenet = ProbenetTrainer(useTpu=useTpu)

    def characterize(self, dataset):
        self.probenet.set_dataset(dataset)
        self.probenet.evaluate()
        return self.probenet.val_acc
