import json, os
from trainers import ProbenetTrainer, ResnetTrainer, StandardTransferLearner
from keras.models import load_model


DATABASE_PATH = 'db.json'


class Database:
    def __init__(self, controller):
        self._db = {}
        self._controller = controller
        self._db[self._controller.name] = {}

    def __getitem__(self, item):
        return self._db[self._controller.name][item]

    def __setitem__(self, item, val):
        self._db[self._controller.name][item] = val

    def __contains__(self, value):
        return value in self._db[self._controller.name]

    def __repr__(self):
        return self._db

    @classmethod
    def load(self, path, controller):
        db_json = None
        with open(path, 'r') as f:
            db_json = json.load(f)
        instance = Database(controller)
        instance._db = db_json
        instance._controller = controller
        if instance._controller.name not in instance._db:
            instance._db[instance._controller.name] = {}
        return instance

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self._db, f, indent=4)

    def get_nearest_neighbour(self, dcn):
        min_key, min_diff = None, 1
        for key, value in self._db[self._controller.name].items():
            dcn_value = value[0]
            if abs(dcn_value - dcn) < min_diff:
                min_key = key
                min_diff = abs(dcn_value - dcn)
        return min_key


class Controller:
    def __init__(self, datasets, characterizer, trainer):
        self.characterizer = characterizer
        self.trainer = trainer

        if self._db_exists:
            self.db = Database.load(DATABASE_PATH, self)
            datasets = [dataset for dataset in datasets if dataset.name not in self.db]
        else:
            self.db = Database(self)
        self.populate_database(datasets)

    @property
    def _db_exists(self):
        return os.path.isfile(DATABASE_PATH)

    @property
    def name(self):
        return type(self.characterizer).__name__ + '_' + type(self.trainer).__name__

    def populate_database(self, datasets):
        for datasetClass in datasets:
            dataset = datasetClass()
            print('Probing %s dataset' % dataset.name)
            dataset_character = self.characterizer.characterize(dataset)
            print('Finding optimal Model for %s dataset' % dataset.name)
            self.trainer.set_dataset(dataset)
            self.trainer.evaluate()
            self.db[dataset.name] = (dataset_character, self.trainer.best_model_path)
            self.db.save(DATABASE_PATH)

    def find_model(self, dataset, transfer_learner):
        print('Probing %s dataset' % dataset.name)
        dataset_character = self.characterizer.characterize(dataset)
        nearest_dataset = self.db.get_nearest_neighbour(dataset_character)
        print('Found nearest_dataset:', nearest_dataset)
        nearest_dataset_character, best_model_path = self.db[nearest_dataset]
        transfer_learner.set_dataset(dataset)
        transfer_learner.transfer_from_model(best_model_path)
        transfer_learner.evaluate()
        self.db[dataset.name] = (dataset_character, transfer_learner.best_model_path)
        self.db.save(DATABASE_PATH)
        # TODO: ability to create trainer from keras model
        return transfer_learner
