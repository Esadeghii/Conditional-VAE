import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



file_path = 'seq_with_dis.csv'
output_model_path = 'model/cvae_model.pt'
output_images_path = 'images'
output_npz_path = 'results'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def ensure_directory_exists(path):
    """Ensure directory exists; if not, create it."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Ensure directories for model, logs, and images exist
ensure_directory_exists(os.path.dirname(output_model_path))
ensure_directory_exists(output_images_path)
ensure_directory_exists(output_npz_path)

# Dataset
class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def oneHotter(seqs):
    charToInt = {"A": 0, "C": 1, "G": 2, "T": 3}
    return [charToInt[let] for let in seqs]

def load_data_set_seq2seq(file_path, batch_size_=16, split=.2, ynormfac=22., verbose=False):
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=True)

    seqs = df.Sequence.values
    lengths = [len(s) for s in seqs]

    min_length_measured = min(lengths)
    max_length_measured = max(lengths)

    condLen = 64
    disColnames = [" dis " + str(i) for i in range(0, condLen)]

    # y_data = np.array(df[disColnames])
    y_data = df[disColnames]
    # y_data = y_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # y_data = y_data.apply(lambda x: (x) / (x.max()))
    y_data = np.array(y_data)
    y_data = y_data/y_data.max(axis=1)[:,None]
    print(" ********    dddd    ", np.max(y_data.max(axis=1)))
    ##
    X = np.array(df.Sequence.apply(oneHotter).tolist())

    print(f"X shape: {X.shape}")
    print(f"y_data shape: {y_data.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_data, test_size=split)#, random_state=235)

    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(),
                                      torch.from_numpy(y_train).float() / ynormfac)

    val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float() / ynormfac)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_)

    return train_loader, val_loader

# Model Base Class
class Model(nn.Module):
    def __init__(self, filepath=None):
        super(Model, self).__init__()
        self.filepath = filepath or 'default_model.pt'
        self.trainer_config = ''

    def forward(self, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def update_filepath(self):
        if not self.filepath:
            raise ValueError("Filepath not initialized")
        cur_dir = os.getcwd()
        self.filepath = os.path.join(cur_dir, 'models', self.__repr__(), self.__repr__() + '.pt')

    def update_trainer_config(self, config):
        self.trainer_config = config
        self.update_filepath()

    def save(self, hidden_size, emb_dim, dropout):
        save_dir = os.path.dirname(self.filepath)
        ensure_directory_exists(save_dir)
        # torch.save(self.state_dict(), self.filepath)
        torch.save({
            'hidden_size': hidden_size,
            'emb_dim': emb_dim,
            'dropout': dropout,
            'model_state_dict': self.state_dict()
            }, self.filepath)
        print(f'Model {self.__repr__()} saved')

    def save_checkpoint(self, epoch_num):
        filename = os.path.join(os.path.dirname(self.filepath), self.__repr__() + '_' + str(epoch_num) + '.pt')
        ensure_directory_exists(os.path.dirname(filename))
        torch.save(self.state_dict(), filename)

    def load(self, state_dict=None, cpu=False):
        if state_dict:
            self.load_state_dict(state_dict)
        else:
            if cpu:
                self.load_state_dict(torch.load(self.filepath, map_location=lambda storage, loc: storage))
            else:
                # self.load_state_dict(torch.load(self.filepath))
                self.load_state_dict(torch.load(self.filepath))

    def xavier_initialization(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

# Conditional VAE Model
class ConditionalSequenceModel(Model):
    ALPHABET = ['A', 'C', 'G', 'T']

    def __init__(self, n_chars=4, seq_len=10, cond_size=64, bidirectional=True, batch_size=32, hidden_layers=1, hidden_size=32, lin_dim=16, emb_dim=10, dropout=0.1):
        super(ConditionalSequenceModel, self).__init__(output_model_path)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.lin_dim = lin_dim
        self.batch_size = batch_size
        self.cond_size = cond_size
        self.beta = 0.002
        self.capacity = 0.0

        # self.emb_lstm = nn.LSTM(input_size=n_chars + cond_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True, dropout=dropout if hidden_layers > 1 else 0, bidirectional=bidirectional)
        self.emb_lstm = nn.LSTM(input_size=n_chars, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True, dropout=dropout if hidden_layers > 1 else 0, bidirectional=bidirectional)
        # self.emb_lstm = nn.LSTM(input_size=n_chars, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True, dropout=0, bidirectional=bidirectional)
        self.latent_linear = nn.Sequential(
            nn.Linear(hidden_size * seq_len * 2 + cond_size, lin_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.latent_mean = nn.Linear(lin_dim, emb_dim)
        self.latent_log_std = nn.Linear(lin_dim, emb_dim)
        # self.latent_mean = nn.Sequential(nn.Linear(lin_dim, emb_dim), nn.Tanh())
        # self.latent_log_std = nn.Sequential(nn.Linear(lin_dim, emb_dim), nn.Tanh())
        self.dec_lin = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            # nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.dec_lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=hidden_layers, dropout=dropout if hidden_layers > 1 else 0, batch_first=True, bidirectional=bidirectional)
        # self.dec_lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=hidden_layers, dropout=0, batch_first=True, bidirectional=bidirectional)
        self.dec_final = nn.Linear(hidden_size * seq_len * 2 + cond_size, n_chars * seq_len)

        self.xavier_initialization()

    def encode(self, x, c):
        # print("encoder c   ", c)
        x_one_hot = x.float().unsqueeze(2)
        x_one_hot = torch.nn.functional.one_hot(x_one_hot.long(), num_classes=self.n_chars).float()
        x_one_hot = x_one_hot.squeeze(2)
        # print("x   ", x_one_hot.shape)
        # print("c 1    ", c.shape)
        # c = c.unsqueeze(1).repeat(1, x.size(1), 1)
        # c = torch.zeros(c.shape).to(device)
        # print("c 2    ", c.shape)
        # xc = torch.cat((x_one_hot, c), dim=-1)
        # print("xc    ", xc.shape)
        # xc = x_one_hot
        hidden, _ = self.emb_lstm(x_one_hot)
        hidden = torch.flatten(hidden, 1)
        # print("hidden    ", hidden.shape)
        # print("c    ", c.shape)
        # print("hidden    ", hidden)
        # print("c    ", c[0,:])
        # print("c    ", c[1,:])
        # print("c    ", c[2,:])
        # c = torch.zeros(c.shape).to(device)
        hidden = torch.cat((hidden, c), dim=-1)
        hidden = self.latent_linear(hidden)
        
        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        # print("mean   ", z_mean)
        # print("std    ", z_log_std)
        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std)), z_mean, z_log_std

    def decode(self, z, c):
        # c = torch.zeros(c.shape).to(device)
        # print("dedcode   ")
        # print("z   ", z.shape)
        # print("c   ", c.shape)
        # zc = torch.cat((z, c), dim=1)
        # zc = z
        hidden = self.dec_lin(z)
        hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        hidden, _ = self.dec_lstm(hidden)
        conc = torch.cat((torch.flatten(hidden, 1), c), dim=1)
        out = self.dec_final(conc)
        return out.view(-1, self.seq_len, self.n_chars)

    def forward(self, x, c):
        dist, z_mean, z_log_std = self.encode(x, c)
        # z = dist.rsample()
        z, prior_sample, prior = self.reparametrize(dist)
        dec = self.decode(z, c)
        return dec, z_mean, z_log_std

    def generate(self, c, nz=50):
        num = c.shape[0]
        z = torch.randn([num, nz]).to(device)
        rec_x = self.decode(z, c=c)
        return rec_x
        # pass
    
    
    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        prior_sample = prior.sample()
        return sample, prior_sample, prior


    def __repr__(self):
        return "ConditionalSequenceModel"

    @staticmethod
    def mean_crossentropy_loss(weights, targets):
        criteria = nn.CrossEntropyLoss(reduction='mean')
        batch_size, seq_len, num_notes = weights.size()
        assert (batch_size == targets.size(0))
        assert (seq_len == targets.size(1))
        weights = weights.view(-1, num_notes)
        targets = targets.view(-1)
        loss = criteria(weights, targets.long())
        return loss

    @staticmethod
    def mean_accuracy(weights, targets):
        _, max_indices = weights.max(2)
        targets = targets.argmax(dim=2) if targets.dim() > 2 else targets
        correct = max_indices == targets
        # print("weights    ", weights.shape)
        # print("max_indices    ", max_indices.shape)
        # print("targets    ", targets.shape)
        # print("numel    ", targets.numel())
        return torch.sum(correct.float()) / targets.numel()

    @staticmethod
    def reconstruction_loss(x, x_recons):
        return ConditionalSequenceModel.mean_crossentropy_loss(weights=x_recons, targets=x)

    @staticmethod
    def kld_gaussian(mu, log_std):
        return -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp(), dim=1)

    def total_loss(self, x, c, recons, dist, mu, log_std):
        recons_loss = self.reconstruction_loss(x, recons)
        kld_loss = torch.mean(self.kld_gaussian(mu, log_std))
        return recons_loss + self.beta * torch.abs(kld_loss - self.capacity)#, recons_loss, kld_loss
    

### load model
# model = ConditionalSequenceModel(n_chars=input_size, seq_len=seq_len, cond_size=condition_size, hidden_size=hidden_size, emb_dim=50, dropout=0.1)
# model.load()
def load_checkpoint(model_path):
    checkpoint = torch.load(model_path)
    # print(checkpoint)
    hidden_size = checkpoint['hidden_size']
    emb_dim = checkpoint['emb_dim']
    dpout = checkpoint['dropout']
    model = ConditionalSequenceModel(n_chars=input_size, seq_len=seq_len, cond_size=condition_size, hidden_size=hidden_size, emb_dim=emb_dim, dropout=dpout)
    model.load(checkpoint['model_state_dict'])
    model = model.to(device)
    return model

# model = load_checkpoint(output_model_path)



input_size = 4
seq_len = 10
condition_size = 64
# load_path = output_model_path
load_path = 'model/cvae_model.pt'
model = load_checkpoint(load_path)



def revOneHotter(seqs):
    intToChar = {0: "A", 1: "C", 2: "G", 3: "T"}
    ret = [[intToChar[let] for let in seq] for seq in seqs]
    return np.array(ret)
def sample_cvae(conditions):
    rec = model.generate(conditions)
    # print(rec.shape)
    labels = torch.argmax(rec, 2)
    # print(labels.shape)
    labels = labels.detach().numpy().tolist()
    ret = revOneHotter(labels)
    # print("shape   ", ret.shape)
    # print(ret)
    return ret

# sampels = sample_cvae(c)

def remove_duplicates(org, seq):
    # print("org   ", org)
    ret = []
    for i in range(len(seq)):
        if seq[i] not in org:
            ret.append(seq[i])
    ret = list(set(ret))
    return ret

def sample(num):#, cls):
    # df2 = df[df['Label'] == cls]
    df = pd.read_csv("seq_with_dis.csv")
    df = df[df['Peak NIR WAV'].notnull()]
    disColnames = [" dis " + str(i) for i in range(0, condition_size)]
    y_data = df[disColnames]
    y_data = np.array(y_data)
    y_data = y_data/y_data.max(axis=1)[:,None]
    ret = []
    for i in range(num):
        rint = np.random.randint(len(y_data))
        ret.append(y_data[rint,:])
    ret = np.array(ret)
    ret = torch.from_numpy(ret).float()
    # print("ret    ", ret.shape)
    generated_samples = sample_cvae(ret)
    # print("samples    ", generated_samples.shape)
    # print("samples    ", type(generated_samples))
    gen_seq = [''.join(row) for row in generated_samples]
    # print("seq    ", gen_seq)
    filtered_seq = remove_duplicates(df['Sequence'].values, gen_seq)
    return filtered_seq

generated_samples = sample(10000)
print("generated samples    ", generated_samples)
print("length    ", len(generated_samples))