
import pandas as pd
from statistics import NormalDist
import numpy as np


def disMaker(center,peakB,condLen):
    # eneergy starts and ends
    eE = 1e9 * (4.135667516E-15) * (299792458) / 400
    eS = 1e9 * (4.135667516E-15) * (299792458) / 1200
    eE = int(eE)
    eS = int(eS)
    mean = 1e9 * (4.135667516E-15) * (299792458) / center

    # standard devation haa coded
    std = 1e9 * (4.135667516E-15) * (299792458) / 10
    # 10/800 = 0.0125 -> x/2 = 0.0125  -> x = .023
    std = .023

    points = np.linspace(eS, eE,condLen+1)
    probVals = np.zeros(condLen+1)

    dis = NormalDist(mean, std)
    s = dis.cdf(points[0])
    # caculates the probability for each bin
    for i in range(1,len(points)):
        e = dis.cdf(points[i])
        prob = e-s
        probVals[i-1] = probVals[i-1] +prob
        s = e

    probVals[-1]=peakB

    return probVals

''' 
Sampling 
'''

file_path = "../data/seq_with_dis.csv"


df = pd.read_csv(file_path)

allBrightness = ['Peak 1 a','Peak 2 a','Peak 3 a','NIR a']
brightness = df[allBrightness ].to_numpy().flatten()
maxB =max(brightness )

condLen = 64


center=  500 # Green
distr = disMaker(center,maxB,condLen)
fname = "samped_DNA/" + "green_dis"
np.save("../" + fname,distr)

center=  620 # Red
distr = disMaker(center,maxB,condLen)
fname = "samped_DNA/" + "red_dis"
np.save("../" + fname,distr)

center=  670 # far Red
distr = disMaker(center,maxB,condLen)
fname = "samped_DNA/" + "far_red_dis"
np.save("../" + fname,distr)

center=  950 # nir
distr = disMaker(center,maxB,condLen)
fname = "samped_DNA/" + "nir_dis"
np.save("../" + fname,distr)

center=  1100 # nir 2
distr = disMaker(center,maxB,condLen)
fname = "samped_DNA/" + "nir_2_dis"
np.save("../" + fname,distr)
