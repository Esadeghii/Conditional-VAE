
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


def load_data_set_seq2seq(file_path,
                          min_length=0, max_length=100, batch_size_=4, output_dim=1, maxdata=9999999999999,
                          tokenizer_y=None, split=.2, ynormfac = 22.,verbose = False ):


    min_length_measured = 0
    max_length_measured = 999999
    protein_df = pd.read_csv(file_path)

    if verbose:
        protein_df.describe()

    df_isnull = pd.DataFrame(
        round((protein_df.isnull().sum().sort_values(ascending=False) / protein_df.shape[0]) * 100, 1)).reset_index()
    df_isnull.columns = ['Columns', '% of Missing Data']
    df_isnull.style.format({'% of Missing Data': lambda x: '{:.1%}'.format(abs(x))})
    cm = sns.light_palette("skyblue", as_cmap=True)
    df_isnull = df_isnull.style.background_gradient(cmap=cm)

    protein_df.drop(protein_df[protein_df['Seq_Len'] > max_length - 2].index, inplace=True)
    protein_df.drop(protein_df[protein_df['Seq_Len'] < min_length].index, inplace=True)

    if verbose:
        print(protein_df.shape)
        print(protein_df.head(6))

    protein_df = protein_df.reset_index(drop=True)

    seqs = protein_df.Sequence.values

    test_seqs = seqs[:1]

    lengths = [len(s) for s in seqs]

    if verbose:
        print(protein_df.shape)
        print(protein_df.head(6))

    min_length_measured = min(lengths)
    max_length_measured = max(lengths)

    if verbose:
        print(min_length_measured, max_length_measured)

    # fig_handle = sns.distplot(lengths,bins=50,kde=False, rug=False,norm_hist=False,axlabel='Length')
    # fig = fig_handle.get_figure()
    # plt.show()
    #
    # INPUT - X
    X = []

    for i in range(len(seqs)):
        X.append([protein_df['AH'][i], protein_df['BS'][i], protein_df['T'][i],
                  protein_df['UNSTRUCTURED'][i], protein_df['BETABRIDGE'][i], protein_df['310HELIX'][i],
                  protein_df['PIHELIX'][i],
                  protein_df['BEND'][i]])

    X = np.array(X)

    if verbose:
        print("sample X data", X[0])
        fig_handle = sns.distplot(X[:,0],bins=50,kde=False, rug=False,norm_hist=False,axlabel='AH')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,1],bins=50,kde=False, rug=False,norm_hist=False,axlabel='BS')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,2],bins=50,kde=False, rug=False,norm_hist=False,axlabel='T')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,3],bins=50,kde=False, rug=False,norm_hist=False,axlabel='UNSTRUCTURED')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,4],bins=50,kde=False, rug=False,norm_hist=False,axlabel='BETABRIDGE')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,5],bins=50,kde=False, rug=False,norm_hist=False,axlabel='310HELIX')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,6],bins=50,kde=False, rug=False,norm_hist=False,axlabel='PIHELIX')
        fig = fig_handle.get_figure()
        plt.show()
        fig_handle = sns.distplot(X[:,7],bins=50,kde=False, rug=False,norm_hist=False,axlabel='BEND')
        fig = fig_handle.get_figure()
        plt.show()

    seqs = protein_df.Sequence.values

    # create and fit tokenizer for AA sequences
    if tokenizer_y == None:
        tokenizer_y = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?[\\]^_`{|}\t\n')
        tokenizer_y.fit_on_texts(seqs)

    y_data = tokenizer_y.texts_to_sequences(seqs)

    y_data = sequence.pad_sequences(y_data, maxlen=max_length, padding='post', truncating='post')

    print("#################################")
    print("DICTIONARY y_data")
    dictt = tokenizer_y.get_config()
    print(dictt)
    num_words = len(tokenizer_y.word_index) + 1

    print("################## max token: ", num_words)

    # revere
    print("TEST REVERSE: ")
    print("y data shape: ", y_data.shape)
    y_data_reversed = tokenizer_y.sequences_to_texts(y_data)

    for iii in range(len(y_data_reversed)):
        y_data_reversed[iii] = y_data_reversed[iii].upper().strip().replace(" ", "")

    print("Element 0", y_data_reversed[0])

    print("Number of y samples", len(y_data_reversed))
    print("Original: ", y_data[:3, :])

    print("REVERSED TEXT 0..2: ", y_data_reversed[0:3])

    print("Len 0 as example: ", len(y_data_reversed[0]))
    # print ("Len 2 as example: ", len (y_data_reversed[2]) )

    if maxdata < y_data.shape[0]:
        print('select subset...', maxdata)
        X = X[:maxdata]
        y_data = y_data[:maxdata]
        print("new shapes: ", X.shape, y_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=split, random_state=235)

    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(y_train).float() / ynormfac)  # /ynormfac)

    # fig_handle = sns.distplot(torch.from_numpy(y_train)*ynormfac,bins=25,kde=False, rug=False,norm_hist=False,axlabel='y labels')
    # fig = fig_handle.get_figure()
    # plt.show()

    test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float() / ynormfac)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
    train_loader_noshuffle = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_)

    return train_loader, train_loader_noshuffle, test_loader, tokenizer_y