import  numpy as np


seed = 203
numRand = 10000

np.random.seed(seed)
randomNums = np.random.rand(numRand)

np.save("../data/seeds.npy",randomNums)

print("end")