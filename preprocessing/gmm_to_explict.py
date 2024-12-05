import pandas as pd
import numpy as np
from statistics import NormalDist
import matplotlib.pyplot as plt

def disToVals(oneDis,probVals,points):


    co =   oneDis.iloc[0]
    mean = oneDis.iloc[1]
    std =  oneDis.iloc[2]

    dis = NormalDist(mean,std)


    s = dis.cdf(points[0])
    # caculates the probability for each bin
    for i in range(1,len(points)):
        e = dis.cdf(points[i])
        prob = e-s
        probVals[i-1] = probVals[i-1] + prob*co
        s = e

    return probVals


def disMaker(oneRow,condLen,avgStd):
    # eneergy starts and ends
    eE = 1e9 * (4.135667516E-15) * (299792458) / 400
    eS = 1e9 * (4.135667516E-15) * (299792458) / 1200
    eE = int(eE)
    eS = int(eS)



    points = np.linspace(eS, eE,condLen+1)

    peaks = oneRow[['Sequence',
                    'Peak 1 a', 'Peak 1 b', 'Peak 1 c', 'Peak 1 b WAV',
                    'Peak 2 a', 'Peak 2 b', 'Peak 2 c', 'Peak 2 b WAV',
                    'Peak 3 a', 'Peak 3 b', 'Peak 3 c', 'Peak 3 b WAV',
                    'NIR a', 'Peak NIR b', 'Peak NIR c', 'Peak NIR WAV']]

    peaks.dropna()

    probVals = np.zeros(condLen )

    disSets =  [['Peak 1 a', 'Peak 1 b', 'Peak 1 c', 'Peak 1 b WAV'],
                ['Peak 2 a', 'Peak 2 b', 'Peak 2 c', 'Peak 2 b WAV'],
                ['Peak 3 a', 'Peak 3 b', 'Peak 3 c', 'Peak 3 b WAV'],
                ['NIR a', 'Peak NIR b', 'Peak NIR c', 'Peak NIR WAV']]

    for disName  in disSets:
        oneDis = peaks[disName]
        # if the dis isn't nan and the sigma is not zero
        if not np.isnan(oneDis.iloc[0]) and  oneDis.iloc[2]!=0:
            # if its NIR hardcode STD
            if 'NIR' in disName[0]:
                oneDis.iloc[2]= avgStd
            probVals = disToVals(oneDis, probVals, points)

    # normalizing and adding a vec
    disMax = max(probVals)+1

    if disMax ==0:
        return np.append(probVals,disMax)
    else:
        normalizedProbs = probVals/disMax
        return np.append(normalizedProbs,disMax)

fpath  = "../data/combined_null_and_dis.csv"

data = pd.read_csv(fpath)

# get average ST
stds = np.asarray(data[['Peak 1 c', 'Peak 2 c', 'Peak 3 c']].to_numpy()).flatten()
avgStd = np.mean(stds[~np.isnan(stds)])



condLen = 64
# gets an array of the dis
disArray =  np.array(data.apply(disMaker,args=(condLen,avgStd), axis=1).tolist())

disColnames = [ " dis " + str(i) for i in range(0,condLen)]
disColnames.append("maxBright")

dis = pd.DataFrame(disArray, columns=disColnames)
data = pd.concat([data, dis], axis=1)

bLog =  np.log(data["maxBright"])

bins = np.linspace(0,max(bLog),64 )
# remove bottom boundry
bins = bins[1:]
# remove top boundry
bins = bins[:-1]

np.savetxt("../data/log_bins.csv", bins, delimiter=",")

hist = bLog.hist(bins=128)


plt.show()

data.to_csv("../data/seq_with_dis.csv")





oneRow  =data.iloc[0]
xvs =  np.array(range(0,condLen))

print(oneRow['Sequence'])
plt.plot(xvs,oneRow[disColnames[:-1]])
plt.show()

print("end")