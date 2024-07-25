from matplotlib.pyplot import axis
import torch
import numpy as np
import pandas as pd
from sequenceModel import SequenceModel
import sequenceDataset as sd
import time
import os
import scipy.stats as stats
from sklearn.decomposition import PCA
import sys
import json
import filter_sampled_sequences as filt

# import sys
# sys.path.append('/Users/matthewkilleen/miniconda3/envs/kdd-sub/bin')


def process_data_file(path_to_dataset: str, sequence_length=10, prepended_name="processed", path_to_put=None, return_path=False):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    label = np.array(data.dataset['Label'])

    if path_to_put is not None:
        file_path = f"{path_to_put}/{prepended_name}-{time.time()}.npz"
    else:
        file_path = f"./data-for-sampling/processed-data-files/{prepended_name}-{time.time()}.npz"

    np.savez(file_path, label=label, ohe=ohe_sequences)
    if return_path:
        return file_path
    


def encode_data(ohe_sequences: object, model: object):
    """This is a wrapper function for the encode() function that can be found in sequenceModel.py. This simply calls
    that function and returns the latent distribution that is produced in the latent space."""
    ohe_sequences = torch.from_numpy(ohe_sequences)
    latent_dist = model.encode(ohe_sequences)
    return latent_dist

def calcuting_cluster_means_and_cluster_stddevs(latent_dist: object, lables: object):
    cluster_means, cluster_stddevs = [],[]
    lables = torch.from_numpy(lables)
    means = latent_dist.mean.detach().numpy()
    stddev = latent_dist.stddev.detach().numpy()
    for i in set(lables.detach().numpy()):
        cluster_means.append(np.mean([idx[1] for idx in zip(lables.detach().numpy(),means) if idx[0] == i],axis=0))
        cluster_stddevs.append(np.mean([idx[1] for idx in zip(lables.detach().numpy(),stddev) if idx[0] == i],axis=0))
    return cluster_means, cluster_stddevs




def sample_from_vae_with_rejection(z_sample, cluster_means, cluster_stddevs, desired_label, num_samples=10000, threshold=1.0):
    """
    Samples from the VAE's latent space such that the probability of the points 
    being from the desired cluster is significantly higher than from other clusters.

    Parameters:
    - vae_model: The trained VAE model.
    - cluster_means: A list of mean vectors for each cluster in the latent space.
    - cluster_stddevs: A list of standard deviation vectors for each cluster in the latent space.
    - desired_label: The label of the desired cluster (0 to 5).
    - num_samples: Number of samples to generate.
    - threshold: The factor by which the probability for the desired cluster should exceed others.

    Returns:
    - accepted_samples: Generated samples from the desired cluster.
    """
    
    Generated_sample = []
    p_desireds = []
    p_others = []
    desired_mean = cluster_means[desired_label]
    desired_stddev = cluster_stddevs[desired_label]
    num_accepted_samples = 0

    while num_accepted_samples < num_samples:
        z_sample = np.random.normal(desired_mean, desired_stddev)
        # Compute the probability density for the desired cluster
        p_desired = stats.multivariate_normal.pdf(z_sample, mean=desired_mean, cov=np.diag(desired_stddev**2))
        
        # Compute the maximum probability density for all other clusters
        max_p_others = 0
        for i in range(len(cluster_means)):
            if i != desired_label:
                mean = cluster_means[i]
                stddev = cluster_stddevs[i]
                p_other = stats.multivariate_normal.pdf(z_sample, mean=mean, cov=np.diag(stddev**2))
                if np.average(p_other) > np.average(max_p_others): # p_other not a number is array
                    max_p_others = p_other
        
        # Accept the sample if the desired probability is significantly higher
        if np.average(p_desired) > threshold * np.average(max_p_others):
            z_sample_tensor = torch.tensor(z_sample, dtype=torch.float32).unsqueeze(0)
            Generated_sample.append(z_sample_tensor)
            p_desireds.append(np.average(p_desired))
            p_others.append(np.average(max_p_others))
            num_accepted_samples += 1
            print(f'{num_accepted_samples} are generated', end='\r')
    return torch.cat(Generated_sample), p_desireds, p_others


def decode_data(z_sample, model):
    """This is a wrapper function for the decode() function found in sequenceModel.py. This takes as input the
    calculated z_sample and returns the decoded sample in the form of a 10 x 4 matrix, where each 4-element array
    represents the numerical estimates for each base in the DNA sequence"""
    decoded_sample = model.decode(z_sample)
    return decoded_sample


def convert_sample(decoded_sample):
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence_length = decoded_sample.shape[0]
    result = ""
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        result += sequence_alphabet.get(max_index)
    return result


def convert_and_write_sample(decoded_sample, f: str):
    """This function takes in the decoded sample returned from decode_data() and takes the maximum value for each base,
    wherein the maximum estimate is replaced by the corresponding base in the DNA sequence. This is written to a csv."""
    sequence_alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequence_length = decoded_sample.shape[0]
    for i in range(sequence_length):
        max_index = np.argmax(decoded_sample[i, :])
        f.write(sequence_alphabet.get(max_index))
    f.write("\n")


def unpack_and_load_data(path_to_file: str, path_to_model: str):
    """This function is used as a wrapper function to load the .npz file that stores the wavelength, LII and one
    hot encoded arrays and load the trained model used for sampling. Both the data file and model are returned
    as objects."""
    data_file = np.load(path_to_file)
    model = torch.load(path_to_model)
    return data_file, model


def compare_sequences(sequence_a, sequence_b):
    """Compares two sequences base by base and returns the ratio of how many they have in common"""
    length = max(len(sequence_a), len(sequence_b))
    num_match = 0
    for i in range(length):
        num_match = num_match + 1 if sequence_a[i] == sequence_b[i] else num_match
    return eval(f"{num_match}/{length}")


def write_detailed_sequences(path_to_put_folder, path_to_sequences, z_label):
    detailed_path = f"{path_to_put_folder}/detailed-sequences"
    with open(detailed_path, 'w') as f:
        with open(path_to_sequences, 'r') as original:
            file_contents = original.readlines()

            newline = "\n".join("")

            initial_line = file_contents[0]
            f.write(initial_line)
            file_contents = file_contents[1:]

            for i, line in enumerate(file_contents):
                if i < np.shape(z_label)[0]:
                    f.write(f"{line[:line.rindex(newline)-1]},{z_label[i]}\n")

    return detailed_path


def write_encoded_sequence_wavelength_lii(path_to_generated: str, path_to_data_file: str,orginal_latent_dist, model,p_desireds,p_others):
    data_file = np.load(path_to_data_file)

    label_array = data_file['label']
    ohe_sequences_tensor = data_file['ohe']

    latent_dist = encode_data(ohe_sequences_tensor, model)
    calcuting_cluster_means_and_cluster_stddevs(latent_dist,lables=label_array)
    mean_matrix = latent_dist.mean.detach().numpy()

    pca = PCA(n_components=1)  # Reduce to 3 dimensions
    principalComponents = pca.fit_transform(mean_matrix)
    z_value = principalComponents[:,0]

    random_sample, _, _ = SequenceModel.reparametrize(model, latent_dist)

    decoded = decode_data(random_sample, model)

    newline = "\n".join("")

    with open(path_to_generated, 'r+') as f:
        file_contents = f.readlines()
        file_contents = file_contents[1:]
        f.truncate(0)

    with open(path_to_generated, 'r+') as f:
        f.write("Sequence Generated,Value Generated,z-values,p_desireds,p_others,Sequence Encoded/Decoded,Value Encoded,z'values,Ratio\n")

        decoded = decoded.detach().numpy()
        for i, line in enumerate(file_contents):
            if i < np.shape(z_value)[0]:
                sequence_original = line.split(',')[0]
                sequence_generated = convert_sample(decoded[i, :, :])
                ratio = compare_sequences(sequence_original, sequence_generated)
                f.write(f"{line[:line.rindex(newline)-1]},{'-'.join([str(lt)for lt in orginal_latent_dist.detach().numpy()[i]])},{p_desireds},{p_others},{sequence_generated},{z_value[i]},{'-'.join([str(lt)for lt in latent_dist.loc.detach().numpy()[i]])},{ratio}\n")


def write_merged_dataset(path_to_base_dataset: str, path_to_generated_samples: str, path_to_put_folder: str):
    generated_file = f"{path_to_put_folder}/merged-data-set"
    #generated_file = f"./data-for-sampling/merged-data-files/{time.time()}.csv"
    with open(generated_file, 'w') as new:
        with open(path_to_base_dataset) as base:
            contents = base.readlines()
            new.writelines(contents)
        with open(path_to_generated_samples) as gen:
            gen.readline()
            contents = gen.readlines()
            new.writelines(contents)

    return generated_file


def sampling(path_to_data_file: str, path_to_model: str, path_to_put: str) -> np.ndarray:
    """This function serves as a main function for the sampling process, taking in the path to the data file with the
    .npz extension, the path to the trained model used for sampling and a path to write the resulting sequences.
    Set the boolean value to True to only return the randomly sampled vectors from the truncated distribution,
    otherwise it decodes samples and writes to file path specified in arguments."""

    path_to_put_folder = f"{path_to_put}/samples-{time.time()}"
    os.mkdir(path_to_put_folder)

    data_file, model = unpack_and_load_data(path_to_data_file, path_to_model)

    label_array = data_file['label']
    ohe_sequences_tensor = data_file['ohe']

    latent_dist = encode_data(ohe_sequences_tensor, model)

    cluster_means, cluster_stddevs = calcuting_cluster_means_and_cluster_stddevs(latent_dist,label_array)

    z_samples,p_desireds,p_others = sample_from_vae_with_rejection(latent_dist, cluster_means, cluster_stddevs, 2, num_samples=10000, threshold=1.0)

    path_to_sequences = f"{path_to_put_folder}/generated-sequences"
    with open(path_to_sequences, 'a', newline='') as f:
        f.write("Sequence,Wavelen,LII\n")
        for i, sample in enumerate(z_samples):
            sample = np.array(sample, dtype='float32')
            sample = torch.tensor(sample)
            decoded_sample = decode_data(sample, model)
            decoded_sample = decoded_sample.detach().numpy()
            decoded_sample = np.reshape(decoded_sample, (decoded_sample.shape[1], decoded_sample.shape[2]))
            convert_and_write_sample(decoded_sample, f)

    post_processing(path_to_sequences, path_to_put_folder, z_samples, model,p_desireds,p_others)


def post_processing(path_to_sequences, path_to_put_folder, z_samples, model,p_desireds,p_others):
    """This is a function that deals with all of the post processing needed in order to filter out repeated sequences that 
    were generated, create annotated files that list out important values in z space and create .npz files necessary to use
    PCA later on"""
    filt.write_unique(path_to_sequences)
    data_set_dict = filt.fill_training_data_dict(sampling_params["Original Data Path"])
    filt.remove_duplicate(data_set_dict, path_to_sequences)

    detailed_data_path = write_detailed_sequences(path_to_put_folder, path_to_sequences, z_samples[:,0])
    generated_data_path = process_data_file(path_to_sequences, prepended_name="generated-sequences-", path_to_put=path_to_put_folder, return_path=True)
    write_encoded_sequence_wavelength_lii(detailed_data_path, generated_data_path,z_samples, model,p_desireds,p_others)

    #merged_path = write_merged_dataset("cleandata.csv", path_to_sequences, path_to_put_folder)
    #process_data_file(merged_path, prepended_name="pca-merged-", path_to_put=path_to_put_folder)
    

if __name__ == "__main__":
    with open("sampling-parameters.json", 'r') as f:
        try:
            data = json.load(f)
            sampling_params = data['Parameters']
        except:
            print("Cannot process parameter file, please make sure sampling-parameters.json is correctly configured.")
            sys.exit(1)
    process_data_file(sampling_params['Original Data Path'], prepended_name='clean-data-base')  # Use this to process the data you wish to use into .npz

    path_to_data_npz = process_data_file(sampling_params['Original Data Path'], prepended_name="clean-data-base", return_path=True)
    sampling(path_to_data_npz, sampling_params['Model Path'], "./data-for-sampling/past-samples-with-info")
    os.remove(path_to_data_npz)
