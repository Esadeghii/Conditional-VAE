
from torch.utils.data import DataLoader, Dataset
import  pandas as pd
import seaborn as sns
import  numpy as np
import matplotlib as plt
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import  torch

class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def oneHotter(seqs):


    # # create and fit tokenizer for AA sequences
    # tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n')
    # tokenizer_y.fit_on_texts(seqs)
    charToInt =  {"A":1,"C":2,"G":3,"T":4}



    # numClasses  =  len(np.unique(catList))
    numClasses = 4
    f= len(seqs)
    ans  = np.array([0]*(f* numClasses) )

    for i,let in enumerate(seqs):
        entry = charToInt[let]
        ans[i*4+entry-1]= 1

    return  ans


def load_data_set_seq2seq(file_path,
                          min_length=0, max_length=10, batch_size_=4, output_dim=1, maxdata=9999999999999,
                          tokenizer_y=None, split=.2, ynormfac = 22.,verbose = False ):

    verbose = True
    min_length_measured = 0
    max_length_measured = 999999
    df = pd.read_csv(file_path)

    if verbose:
        df.describe()


    if verbose:
        print(df.shape)
        print(df.head(6))

    df = df.reset_index(drop=True)

    seqs = df.Sequence.values
    lengths = [len(s) for s in seqs]

    if verbose:
        print(df.shape)
        print(df.head(6))

    min_length_measured = min(lengths)
    max_length_measured = max(lengths)

    if verbose:
        print(min_length_measured, max_length_measured)


    condLen  =64
    disColnames = [" dis " + str(i) for i in range(0, condLen)]
    disColnames.append("maxBright")

    X = np.array(df[ disColnames ])



    # turns nuermic into one hot encoding
    y_data =  np.array( df.Sequence.apply(oneHotter).tolist())

    [samps,cols]= y_data.shape
    y_data = y_data.reshape((samps,int(cols/4),4 ) )
    y_data = np.transpose(y_data,axes=(0,2,1))


    #y_data = sequence.pad_sequences(y_data, maxlen=max_length, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=split, random_state=235)

    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(y_train).float() / ynormfac)  # /ynormfac)

    test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float() / ynormfac)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
    train_loader_noshuffle = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_)

    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y