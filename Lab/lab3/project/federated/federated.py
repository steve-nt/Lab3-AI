import os   # Imported from Stefanos Remove it if not needed
import copy # Imported from Stefanos Remove it if not needed
import random
import tensorflow as tf

from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
from dataset.dataset import Dataset
from federated.math import federated_math

from ood.hdff import Hdff
from model.model import Model

class Federated():
    """
        Federated learning environment. Three cycles per round, update local models from global model, train local models, regression on local models and update global. 
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
        self.dataset = dataset
        self.init_model = model            # This model can be 
        self.federated_config = federated_config
        self.ood_config = ood_config
        self.plot_config = plot_config
        self.dataset_config = dataset_config # Imported from Stefanos Remove it if not needed
    
    def run(self): 
        """
            Runs federated learning environment.
        """
        round = 0
        
        # If loading pretrained models, round must be updated with the pre-trained models round. 
# Imported from Stefanos Remove it if not needed
        if self.federated_config.load:
            # Include global (host_id) and locals
            for client_id in range(self.federated_config.clients + 1): 
                model_path = f"{self.federated_config.path}model{client_id}_round{self.federated_config.load_round}.keras"
                if os.path.exists(model_path):
                    if client_id == self.federated_config.host_id:
                        self.global_model.model = tf.keras.models.load_model(model_path)
                    elif client_id in self.local_models:
                        self.local_models[client_id].model = tf.keras.models.load_model(model_path)
            # Update the round counter to the loaded round
            round = self.federated_config.load_round
            print(f"Loaded models from round {round}")        
# Imported from Stefanos Remove it if not needed


        if(round < int(self.federated_config.rounds)):  
            round = self.train_(round)
        else:                                            # Only test if round = round on specific clients in load. 
            self.test_()
            
        return None
  
    def train_(self, start : int):
        # For training, this will be the functionality flow. However you need to implement them. 
        
        for round in range(1+start, self.federated_config.rounds+1):   
            part = max(int(self.federated_config.participants), 1)                 # Alteast one client will partcipate in round.
            selected_clients = random.sample(list(self.local_models.keys()), part)                           # Imported from Stefanos Remove it if not needed # TODO Select random clients that will participate during training round. 
            
            while self.federated_config.host_id in selected_clients:               # If global model gets selected as participant, select new.
                selected_clients = random.sample(list(self.local_models.key()), part)
            
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

# Imported from Stefanos Remove it if not needed
            if self.federated_config.save:
                # Save global model
                self.global_model.model.save(f"{self.federated_config.path}model{self.federated_config.host_id}_round{round}.keras")
                # Save local models
                for client_id, local_model in self.local_models.items():
                    local_model.model.save(f"{self.federated_config.path}model{client_id}_round{round}.keras")
# Imported from Stefanos Remove it if not needed    
        return round
# Imported from Stefanos Remove it if not needed   
    def test_(self):
        """ Evaluate the trained global model on test data """
        # The global model is assigned the datasets corresponding to its host_id
        dataset_idx = self.federated_config.client_to_dataset[self.federated_config.host_id]
        
        # Unpack the test data
        _, _, test_data = self.dataset.get(dataset_idx)
        
        print("\n--- Testing Global Model ---")
        self.global_model.test(test_data)
        
        # Plot the performance metrics if plotting is enabled
        if self.plot_config.plot:
            self.global_model.plot_all(test_data, xlabel="Global Model", title="Global Model Performance")
        return None
# Imported from Stefanos Remove it if not needed     
    
    def global_(self, id : int, round : int):                                            # Update all local models with global model weights. 
        """
            Updates local models with global model weights. 
        Args:
            id (int): id for global model.
        """
        
# Imported from Stefanos Remove it if not needed   
        # 3.2.2 Regression
        # Get current global weights
        global_weights = self.global_model.get_weights()
        
        # Distribute to all local models
        for client_id in self.local_models.keys():
            self.local_models[client_id].set_weights(global_weights)
# Imported from Stefanos Remove it if not needed   
        return None
            
    def local_(self, id : int, round : int):                                         # Train local models
        """
            Trains local models, with id. 
        Args:
            id (int): id for local model. 
            round (int): current round. 
        """
# Imported from Stefanos Remove it if not needed   
        # 3.2.3 Train
        local_model = self.local_models[id]
        
        # Get the assigned dataset for this client
        dataset_idx = self.federated_config.client_to_dataset[id]
        train_data, val_data, _ = self.dataset.get(dataset_idx)
        
        print(f"--- Client {id} is training locally (Round {round}) ---")
        # Train the local model for the configured number of epochs
        local_model.train(train_data, val_data, epochs=self.federated_config.epochs)
# Imported from Stefanos Remove it if not needed   
        return None
        
    def update_(self, selected_clients, round : int):
        """
            Update global model with clients that was training during round (selected clients).
            
            Incorporate ood detection if enabled in config. Select only clients that are not detected as ood.
        Args:
            selected_clients (list): list with id of clients (local models) that selected for training.
            round (int): current round. 
        """
# Imported from Stefanos Remove it if not needed 
        # TODO
        # 3.2.4 Aggregation
        collected_weights = []
        
        # Collect weights from all clients that trained this round
        for client_id in selected_clients:
            collected_weights.append(self.local_models[client_id].get_weights())
            
        # Federated Averaging (FedAvg) implementation
        if collected_weights:
            new_weights = []
            # Iterate through each hidden layer
            for layer_index in range(len(collected_weights[0])):
                # Collect weights for this specific layer across all participants
                layer_weights = [client[layer_index] for client in collected_weights]
                
                # Compute the mean for this layer
                layer_mean = tf.math.reduce_mean(layer_weights, axis=0)
                new_weights.append(layer_mean)
                
            # Update global model with the newly averaged weights
            self.global_model.set_weights(new_weights)
                
# Imported from Stefanos Remove it if not needed         
        return None
        
        
    def ood_extraction(self, id : int, model : Model):
        """ Exctract features from model. 

        Args:
            id (int): Id of model.
            model (Model): Model (object).
        """
        # TODO
        
        return None
        
    def ood_detection(self, selected_clients):
        """ Detecting model being ood from selected clients that trains.

        Args:
            selected_clients (int): clients that undergo training this round.
        """
        # TODO
        
        return None
            
    def result(self):
        """
            Plot performance of each model. 
        """
        # TODO
        
        return None