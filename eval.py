from controller import Controller
from datasets import *
from trainers import *
from characterizer import *


DATASETS =  [
                # Mnist,
                # FashionMnist,
                Cifar10,
                Cifar100,
                SVHN,
                GTSRB,
                Flowers102,
                # Flowers
            ]

test_dataset = Flowers()

controller = Controller(
                datasets=DATASETS,
                characterizer=ProbenetCharacterizer(),
                trainer=EnasTrainer(),
            )


model = controller.find_model(
            dataset=test_dataset,
            transfer_learner=EnasTransferLearner()
        )

print('Transfer Learning accuracy: ', model.val_acc)
'''
trainer = EnasTrainer(test_dataset)
trainer.evaluate()

print('Manual Training accuracy: ', trainer.val_acc)
'''
'''
Controller is not trained
Cifar100
Transfer Learning accuracy: 0.0975
Manual Training accuracy: 0.0982

Flowers
Transfer Learning accuracy:  0.3378746598877764
Manual Training accuracy:  0.47547683931826246
'''
