import os
import tensorflow as tf
import random
import numpy as np

from dataset.dataset import Dataset
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
from federated.federated import Federated
from model.model import Model
from model.math.plot import ModelPlot

def set_seeds(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

def set_global_determinism(SEED):
    set_seeds(SEED=SEED)

SEED = 42
set_global_determinism(SEED=SEED)

class ModelSimulation():
    def run(self):
        # (Content omitted for brevity. It remains unchanged from original.)
        pass
        
class FederatedSimulation():
    federated_config = ConfigFederated(
        debug = True,                       
        save = True,                        # MUST BE TRUE FOR PHASE 1: Saves pre-trained models
        load_round = 0,                     
        load_reg = True,                    
        load = False,                       
        delete_on_load = False,             
        path = "./.env/.saved/",            
        rounds = 40,                        # Phase 1: Train for 30-40 rounds on ID data
        ood_round = 41,                     
        clients = 5,                        
        participants = 4,                   
        host_id=0,                          
        client_to_dataset=[[0,1,2,3],[0],[1],[2],[3]]   
    )
    
    ood_config = ConfigOod(
        debug = True,                       
        hdc_debug = False,                  
        enabled = False,                    # MUST BE FALSE FOR PHASE 1: Pre-training disables OOD detection
        hyper_size=int(1e4),                
        id_client = [1,2,3,4],              
        ood_client = [5,6],                 
        ood_protection = False,             # MUST BE FALSE FOR PHASE 1
        ood_protection_thres = 0.7          
    )
    
    model_config = ConfigModel(
        debug = True,                       
        epochs = 1,                         
        activation = 'relu',                
        activation_out = 'softmax',         
        optimizer = 'adam',                 
        loss = 'categorical_crossentropy'   
    )
    
    dataset_config = ConfigDataset(
        debug = False,                      
        batch_size = 64,                    
        image_size = 256,                   
        input_shape = (256,256,1),          
        split = 0.25,                       
        number_of_classes = 2               
    )
    
    plot_config = ConfigPlot(
        plot = True,                       
        path = './.env/plot',               
        img_per_class = 10                  
    )
    
    def run(self):
        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        dataset = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),                                        # id 0
                (Btumor3000().ID, Btumor3000(), []),                                        # id 1
                (Balzheimer5100().ID, Balzheimer5100(), []),                                # id 2
                (Lpneumonia5200().ID, Lpneumonia5200(), [])                                 # id 3
            ],
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        ).mergeAll() 
        
        federated = Federated(
            dataset=dataset,                        
            model=m,                                
            federated_config=self.federated_config, 
            ood_config=self.ood_config,             
            dataset_config=self.dataset_config,     
            plot_config=self.plot_config            
        )
        return federated.run()
    
if __name__ == "__main__": 
    # Switch execution context from ModelSimulation to FederatedSimulation
    # sim1 = ModelSimulation()
    # sim1.run()
    
    sim_federated = FederatedSimulation()
    sim_federated.run()