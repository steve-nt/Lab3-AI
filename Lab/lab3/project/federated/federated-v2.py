import os
import copy
import random
import tensorflow as tf

from config import ConfigDataset, ConfigFederated, ConfigOod, ConfigPlot
from dataset.dataset import Dataset
from ood.hdff import Hdff
from model.model import Model

class Federated():
    def __init__(self, dataset : Dataset, model : Model, federated_config : ConfigFederated, ood_config : ConfigOod, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        self.dataset = dataset
        self.federated_config = federated_config
        self.ood_config = ood_config
        self.plot_config = plot_config
        self.dataset_config = dataset_config
        
        # 3.2.1 Initialize: Create distinct copies for the Global Model and Local Models
        self.global_model = copy.deepcopy(model)
        self.local_models = {}
        
        for client_id in range(1, self.federated_config.clients):
            self.local_models[client_id] = copy.deepcopy(model)

    def run(self): 
        round = 0
        if self.federated_config.load:
            for client_id in range(self.federated_config.clients): 
                model_path = f"{self.federated_config.path}model{client_id}_round{self.federated_config.load_round}.keras"
                if os.path.exists(model_path):
                    if client_id == self.federated_config.host_id:
                        self.global_model.model = tf.keras.models.load_model(model_path)
                    elif client_id in self.local_models:
                        self.local_models[client_id].model = tf.keras.models.load_model(model_path)
            round = self.federated_config.load_round
            print(f"Loaded models from round {round}")        

        if(round < int(self.federated_config.rounds)):  
            round = self.train_(round)
        else:                                            
            self.test_()
            
        return None
  
    def train_(self, start : int):
        for round in range(1+start, self.federated_config.rounds+1):   
            part = max(int(self.federated_config.participants), 1)                 
            selected_clients = random.sample(list(self.local_models.keys()), part)                           
            
            while self.federated_config.host_id in selected_clients:               
                selected_clients = random.sample(list(self.local_models.keys()), part)
            
            if(self.ood_config.enabled and round < self.federated_config.ood_round):
                for i in self.ood_config.ood_client:
                    if i in selected_clients:
                        selected_clients.remove(i)
            
            # 3.2.2 Regression
            self.global_(self.federated_config.host_id, round)                     
            
            # 3.2.3 Training
            for id in selected_clients:                                            
                self.local_(id, round)

            # 3.2.4 Aggregation
            self.update_(selected_clients, round)

            # 3.2.6 Saving Models
            if self.federated_config.save:
                self.global_model.model.save(f"{self.federated_config.path}model{self.federated_config.host_id}_round{round}.keras")
                for client_id, local_model in self.local_models.items():
                    local_model.model.save(f"{self.federated_config.path}model{client_id}_round{round}.keras")
                    
        return round
        
    def test_(self):
        dataset_idx = self.federated_config.client_to_dataset[self.federated_config.host_id]
        _, _, test_data = self.dataset.get(dataset_idx)
        print("\n--- Testing Global Model ---")
        self.global_model.test(test_data)
        if self.plot_config.plot:
            self.global_model.plot_all(test_data, xlabel="Global Model", title="Global Model Performance")
        return None
    
    def global_(self, id : int, round : int):                                            
        global_weights = self.global_model.get_weights()
        for client_id in self.local_models.keys():
            self.local_models[client_id].set_weights(global_weights)
        return None
            
    def local_(self, id : int, round : int):                                         
        local_model = self.local_models[id]
        dataset_idx = self.federated_config.client_to_dataset[id]
        train_data, val_data, _ = self.dataset.get(dataset_idx)
        print(f"--- Client {id} is training locally (Round {round}) ---")
        local_model.train(train_data, val_data, epochs=self.federated_config.epochs)
        return None
        
    def update_(self, selected_clients, round : int):
        collected_weights = []
        for client_id in selected_clients:
            collected_weights.append(self.local_models[client_id].get_weights())
            
        if collected_weights:
            new_weights = []
            for layer_index in range(len(collected_weights[0])):
                layer_weights = [client[layer_index] for client in collected_weights]
                layer_mean = tf.math.reduce_mean(layer_weights, axis=0)
                new_weights.append(layer_mean)
            self.global_model.set_weights(new_weights)
        return None
        
    def ood_extraction(self, id : int, model : Model):
        # Ignored in Phase 1
        return None
        
    def ood_detection(self, selected_clients):
        # Ignored in Phase 1
        return None
            
    def result(self):
        return None