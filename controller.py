from database import Database
from probenet import Prober
from datasets import Cifar100

db = Database()

myDataset = Cifar100()
prober = Prober(myDataset)
dcn = prober.evaluate()

nearestDataset = db.get_nearest_neighbour(dcn)

print(nearestDataset)
print(db[nearestDataset])
