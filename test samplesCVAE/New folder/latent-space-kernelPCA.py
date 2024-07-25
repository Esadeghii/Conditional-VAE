from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import pandas as pd
import numpy as np
import torch
import plotly.express as px

def preprocess_pca(z):
    """Preprocessing before PCA visualization can occur (returns needed objects such as principal components, principal
    components dataframe, and explained variance ratio)"""

    # Standardize the data
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)

    # Kernel PCA
    pca = KernelPCA(n_components=4, kernel='rbf')
    principalComponents = pca.fit_transform(z_scaled)
    explained_variance = np.var(principalComponents, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    return principalComponents, principalDf, explained_variance_ratio

def pca_visualize_3D(principalComponents, principalDf, Labels):
    """Visualizes the PCA result in 3D."""
    clusterscolor = {0: 'R-FR', 1: 'G', 2: 'N', 3: 'G-R', 4: 'G-N', 5: 'G-FR'}
    Labels_with_color = [clusterscolor[i] for i in Labels]
    principalDf['Labels'] = Labels_with_color

    # Adjust point size and transparency
    fig = px.scatter_3d(
        principalDf,
        x='PC1',
        y='PC2',
        z='PC3',  # Use the third principal component for the z-axis
        color=Labels_with_color,
        color_discrete_sequence=["#93220a", "green", "yellow", "red", "blue", "black"],
        opacity=0.7,  # Adjust transparency
        size_max=5    # Adjust point size
    )
    fig.show()

def process_data(path_to_dataset):
    """Loads the dataset from a .npz file."""
    data = np.load(path_to_dataset)
    return data

def get_label_arr(data):
    """Retrieves the label array from the dataset."""
    label = data["Labels"]
    return label

def get_ohe_data(data):
    """Retrieves the one-hot encoded data from the dataset and returns it as a PyTorch tensor."""
    ohe_data = torch.from_numpy(data['ohe'])
    return ohe_data

def process_model(path_to_model):
    """Loads a PyTorch model from a .pt file."""
    model = torch.load(path_to_model)
    return model

def get_z_from_latent_distribution(model, ohe_data):
    """Extracts the latent space representation from the model."""
    latent_dist = model.encode(ohe_data)
    z = latent_dist.loc.detach().numpy()
    return z

def conduct_visualizations(path_to_dataset: str, path_to_model):
    """Conducts PCA visualizations using the given dataset and model."""
    data = process_data(path_to_dataset)
    model = process_model(path_to_model)
    ohe_data = get_ohe_data(data)
    z = get_z_from_latent_distribution(model, ohe_data)
    Labels = get_label_arr(data)
    principalComponents, principalDf, explained_variance_ratio = preprocess_pca(z)
    pca_visualize_3D(principalComponents, principalDf, Labels)

# Example of how to use the functions (make sure the paths are correct)
conduct_visualizations('data-for-sampling/processed-data-files/processed-1710555155.767748.npz', 'models/a20lds20b0.007g1d1h10.pt')
