# Import TensorFlow for neural network operations
import tensorflow as tf
# Import numpy for numerical operations
import numpy as np

# Import seaborn for creating enhanced heatmaps and visualizations
import seaborn as sns

# Import matplotlib pyplot for creating and displaying plots
import matplotlib.pyplot as plt
# Import matplotlib ticker for customizing axis labels and formatting
import matplotlib.ticker as mticker

# Import configuration classes for federated learning and OOD detection settings
from config import ConfigFederated, ConfigOod
# Import the Clients class for managing federated learning clients
from federated.client.clients import Clients
# Import the Hdff class (Hyperdimensional Feature Fusion) for OOD detection
from ood.hdff import Hdff

class FederatedPlot:
    # Method to plot OOD detection results for a single model
    def plot_ood(self, hdff: Hdff, federated_config : ConfigFederated, ood_config : ConfigOod, xlabel : str, title : str):
        # Create a new figure with custom title and size (14x5 inches)
        plt.figure(num=title, figsize=(14, 5))
        # Set the x-axis to display only integer values for better readability of rounds
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1)) 
        
        # Plot the OOD results (similarity scores over rounds) from the HDFF object
        # hdff.results contains the similarity scores comparing local model to global model
        plt.plot(hdff.results, label="Local model " + "(" + str(id) + ")")

        # Set the x-axis label with the provided xlabel parameter (e.g., "Round", "Epoch")
        plt.xlabel(xlabel)
        # Set the y-axis label describing what the values represent (cosine similarity)
        plt.ylabel('Cosine (arg), Model vs. Global Model')
        # Set the title combining provided title with the OOD protection mechanism type
        plt.title(title + " (" + "Security mechanism="+str(ood_config.ood_protection) + ")")
        # Draw a horizontal dashed line at the OOD threshold to visualize the detection boundary
        # min/max of xlim ensures the line spans the entire x-axis width
        # 'k--' specifies a black dashed line style
        plt.plot([min(plt.xlim()),max(plt.xlim())],[float(ood_config.ood_protection_thres),float(ood_config.ood_protection_thres)], 'k--', label="OOD threshold")
        # Add a legend to identify the model line and threshold line
        plt.legend()
        
    def plot_ood_dict(self, result : dict, federated_config : ConfigFederated, ood_config : ConfigOod, xlabel : str, title : str):
        # Create a new figure with custom title and size (14x5 inches)
        plt.figure(num=title, figsize=(14, 5))
        # Set the x-axis to display only integer values for better readability of rounds
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1)) 
        # Iterate through all client IDs from 0 to the total number of clients configured
        for id in range(0, federated_config.clients): 
            # Check if the current ID is the host (global model) ID
            if(id == federated_config.host_id):
                # Plot the global model's results with a label indicating it's the global model
                plt.plot(result[id], label="Global model " + "(" + str(id) + ")")
            else:
                # Plot the local model's results with a label indicating it's a local model and its ID
                plt.plot(result[id], label="Local model " + "(" + str(id) + ")")

        # Set the x-axis label with the provided xlabel parameter (e.g., "Round", "Epoch")
        plt.xlabel(xlabel)
        # Set the y-axis label describing what the values represent (cosine similarity)
        plt.ylabel('Cosine (arg), Model vs. Global Model')
        # Set the title combining provided title with the OOD protection mechanism type
        plt.title(title + " (" + "Security mechanism="+str(ood_config.ood_protection) + ")")
        # Draw a horizontal dashed line at the OOD threshold to visualize the detection boundary
        # min/max of xlim ensures the line spans the entire x-axis width
        # 'k--' specifies a black dashed line style
        plt.plot([min(plt.xlim()),max(plt.xlim())],[float(ood_config.ood_protection_thres),float(ood_config.ood_protection_thres)], 'k--', label="OOD threshold")
        # Add a legend to identify each model line (local/global) and the threshold line
        plt.legend()