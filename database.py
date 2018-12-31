import json, os
from datasets import *
from probenet import Prober


DATASETS =  [
                Mnist,
                FashionMnist,
                Cifar10,
                Cifar100
            ]


DATABASE_PATH = 'db.json'


class Database:
    def __init__(self):
        if self._dbExists:
            self.load()
        else:
            self.populate(DATASETS)
            self.save()

    @property
    def _dbExists(self):
        return os.path.isfile(DATABASE_PATH)

    def populate(self, datasets):
        self._db = {}
        for datasetClass in datasets:
            dataset = datasetClass()
            print('Commencing training of %s dataset' % dataset.name)
            prober = Prober(dataset)
            self._db[dataset.name] = prober.evaluate()

    def load(self):
        with open(DATABASE_PATH, 'r') as f:
            self._db = json.load(f)

    def save(self):
        with open(DATABASE_PATH, 'w') as f:
            json.dump(self._db, f, indent=4)

    def get_nearest_neighbour(self, dcn):
        minKey, minDiff = None, 1
        for key, value in self._db.items():
            if abs(value - dcn) < minDiff:
                minKey = key
                minDiff = abs(value - dcn)
        return minKey

    def __getitem__(self, item):
        return self._db[item]
