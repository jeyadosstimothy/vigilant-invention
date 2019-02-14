from controller import Controller
from datasets import *
from trainers import *
from characterizer import *


DATASETS =  [
                Mnist,
                FashionMnist,
                Cifar10,
                SVHN,
                GTSRB,
                Flowers102,
                Flowers
            ]

test_dataset = Cifar100()

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

trainer = EnasTrainer(test_dataset)
trainer.evaluate()

print('Manual Training accuracy: ', trainer.val_acc)
