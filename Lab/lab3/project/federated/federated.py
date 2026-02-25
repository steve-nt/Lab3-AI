# Import TensorFlow library for neural network operations
import tensorflow as tf

# Import the random module for selecting random clients during each training round
import random

# Import configuration classes for dataset, federated learning, OOD detection, and plotting
from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
# Import the custom Dataset class for handling training data
from dataset.dataset import Dataset
# Import the federated_math module for mathematical operations specific to federated learning
from federated.math import federated_math

# Import the Hdff class (Hyperdimensional Feature Fusion) for OOD detection
from ood.hdff import Hdff
# Import the custom Model class for neural network operations
from model.model import Model

class Federated():
    """
        Federated learning environment. Three cycles per round: update local models from global model, train local models, regression on local models and update global.
    """
    def __init__(self, dataset : Dataset, model : Model, federated_config : ConfigFederated, ood_config : ConfigOod, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Args:
            dataset (Dataset): dataset, custom.
            model (Model): nn. model.
            federated_config (ConfigFederated): configuration for federated learning env. 
            ood_config (ConfigFederated): configuration for ood detection.
            dataset_config (ConfigDataset) : configuration for dataset. 
            plot_config (ConfigPlot): configuration for plotting.
        """
        # Store the dataset object for accessing training data
        self.dataset = dataset
        # Store the initial model that will be used as the global model
        # This model can be distributed to clients or trained locally
        self.init_model = model            
        # Store federated learning configuration (rounds, number of clients, etc.)
        self.federated_config = federated_config
        # Store OOD detection configuration (OOD thresholds, protection mechanisms, etc.)
        self.ood_config = ood_config
        # Store plot configuration (figure sizes, display options, etc.)
        self.plot_config = plot_config
    
    def run(self): 
        """
            Runs federated learning environment.
        """
        # Initialize the round counter to zero (starting round for training)
        round = 0
        
        # If loading pretrained models, round must be updated with the pre-trained models round. 

        # Check if the current round is less than the total number of rounds configured
        if(round < int(self.federated_config.rounds)):  
            # Call train_ function to start federated training, starting from current round
            round = self.train_(round)
        else:                                            
            # Only test if round = round on specific clients in load. 
            # Run testing phase (evaluation on test data)
            self.test_()
            
        # Return None as the function completes execution
        return None
  
    def train_(self, start : int):
        # For training, this will be the functionality flow. However you need to implement them. 
        
        for round in range(1+start, self.federated_config.rounds+1):   
            part = max(int(self.federated_config.participants), 1)                 # Alteast one client will partcipate in round.
            selected_clients = random.sample(None, part)                           # TODO Select random clients that will participate during training round. 
            
            while self.federated_config.host_id in selected_clients:               # If global model gets selected as participant, select new.
                selected_clients = random.sample(None, part)
            
            # This is for ood, if ood is enabled and round is less than ood_round, remove clients that are in ood_client list.
            # Can be good to let the models train for 2 rounds before ood, to get a better model.
            # Some warmup of model before ood and ood client included. 
            if(self.ood_config.enabled and round < self.federated_config.ood_round):
                for i in self.ood_config.ood_client:               
                    selected_clients.remove(i)
            
            self.global_(self.federated_config.host_id, round)                     # Update all local models with global model.
            
            for id in selected_clients:                                            # Train all local models. 
                self.local_(id, round)

            self.update_(selected_clients, round)
    
        return round
    
    def test_(self):
        # TODO: Implement testing phase
        # This method should evaluate the trained models on test data
        return None
    
    def global_(self, id : int, round : int):                                            
        # Update all local models with global model weights. 
        """
            Updates local models with global model weights. 
        Args:
            id (int): id for global model.
        """
        # TODO: Implement synchronization of global model weights to all local models
        return None
            
    def local_(self, id : int, round : int):                                         
        # Train local models
        """
            Trains local models, with id. 
        Args:
            id (int): id for local model. 
            round (int): current round. 
        """
        # TODO: Implement local model training for a specific client
        return None
        
    def update_(self, selected_clients, round : int):
        """
            Update global model with clients that was training during round (selected clients).
            
            Incorporate ood detection if enabled in config. Select only clients that are not detected as ood.
        Args:
            selected_clients (list): list with id of clients (local models) that selected for training.
            round (int): current round. 
        """
        # TODO: Implement aggregation of client updates and global model update
        # Include OOD detection filtering if enabled
        return None
        
        
    def ood_extraction(self, id : int, model : Model):
        """ Exctract features from model. 

        Args:
            id (int): Id of model.
            model (Model): Model (object).
        """
        # TODO: Implement feature extraction from model for OOD detection using HDFF
        # Should extract hyperdimensional features that represent the model
        return None
        
    def ood_detection(self, selected_clients):
        """ Detecting model being ood from selected clients that trains.

        Args:
            selected_clients (int): clients that undergo training this round.
        """
        # TODO: Implement OOD detection logic to identify anomalous models among selected clients
        # Compare client models against global model using similarity metrics
        return None
            
    def result(self):
        """
            Plot performance of each model. 
        """
        # TODO: Implement result visualization and plotting of model performance metrics
        return None