# Import operating system utilities for environment and file operations
import os
# Import TensorFlow for deep learning model operations
import tensorflow as tf
# Import random for random number generation and sampling
import random
# Import NumPy for numerical array operations and mathematics
import numpy as np

# Import Dataset class for handling merged/multi-source datasets
from dataset.dataset import Dataset
# Import all configuration classes for different components
from config import ConfigFederated, ConfigOod, ConfigModel, ConfigDataset, ConfigPlot
# Import various dataset classes from Kaggle datasets
from dataset.download.a_faces16000 import Afaces16000
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_alzheimer5100_poisoned import Balzheimer5100_poisoned
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
from dataset.download.l_pneumonia5200 import Lpneumonia5200
# Import Federated class for running federated learning simulations
from federated.federated import Federated
# Import Model class for neural network model creation and training
from model.model import Model

# Import plotting utilities for model visualization
from model.math.plot import ModelPlot

#
# FEEL FREE TO EDIT THE CONTENT OF ALL GIVEN FILES AS YOU LIKE.
#

############# REPRODUCIBILITY, deterministic behavior #############
def set_seeds(SEED):
    """ Set seeds for deterministic behavior across all libraries. 
    """
    # Set Python hash seed for reproducible dictionary/set ordering
    os.environ['PYTHONHASHSEED'] = str(SEED)
    # Set random library seed for reproducible random numbers
    random.seed(SEED)
    # Set TensorFlow random seed for reproducible neural network behavior
    tf.random.set_seed(SEED)
    # Set NumPy random seed for reproducible array operations
    np.random.seed(SEED)

def set_global_determinism(SEED):
    # Call set_seeds to configure all random seeds
    set_seeds(SEED=SEED)
    
    # Uncomment below if limiting cpu threads, may help with determinism. However may slow down training, especially on large models.
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # CUDA/GPU users uncomment below for deterministic GPU operations
    # os.environ['TF_GPU_ALLOCATOR'] = "cuda_malloc_async"
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # tf.config.experimental.enable_op_determinism()

# Set the random seed for reproducible experiments (change as needed)
SEED = 42
# Initialize global determinism with the chosen seed
set_global_determinism(SEED=SEED)
###############################################################

# Class for testing a single neural network model (non-federated)
class ModelSimulation():
    """ Only for testing single model (non-federated learning scenario). 
    """
           
    def run(self):
        # Configure neural network model parameters
        model_config = ConfigModel(
            debug = True,                       # Enable debug output during training
            epochs = 5,                         # Number of training passes through data
            activation = 'relu',                # ReLU activation for hidden layers
            activation_out = 'softmax',         # Softmax activation for output layer (multi-class)
            optimizer = 'adam',                 # Adam optimizer for weight updates
            loss = 'categorical_crossentropy'   # Loss function for classification
        )
        # Configure dataset handling and preprocessing
        dataset_config = ConfigDataset(
            debug = True,                   # DISABLE IF YOU WANT TO PREVENT IMAGE EXAMPLES FROM BEING DISPLAYED BEFORE TRAINING.
            batch_size = 64,                # Process 64 samples per training update
            image_size = 256,               # Resize images to 256x256 pixels
            input_shape = (256,256,1),      # Grayscale images (1 channel) of size 256x256
            split = 0.25,                   # Use 25% of data for testing/validation
            number_of_classes=2             # Binary classification (2 classes)
        )
        # Configure visualization and plotting
        plot_config = ConfigPlot(
            plot = True,                    # Enable plot generation and display
            path = "./.env/.saved/",        # Directory for saving plots
            img_per_class = 10              # Show 10 sample images per class
        )
        
        # Create the neural network model instance
        m = Model(
            model_config=model_config,
            dataset_config=dataset_config,
            plot_config=plot_config
        )
        
        # Create and merge datasets from multiple sources
        train_data, validation_data, test_data = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),        # Brain tumor dataset 4600 samples
                (Btumor3000().ID, Btumor3000(), []),        # Brain tumor dataset 3000 samples
                (Balzheimer5100().ID, Balzheimer5100(), []), # Alzheimer's dataset 5100 samples
                (Lpneumonia5200().ID, Lpneumonia5200(), []) # Pneumonia dataset 5200 samples
            ],
            dataset_config=dataset_config,
            plot_config=plot_config
        ).mergeAll()  # Merge all datasets into single train/val/test split
        
        # Below is an example of dataset with subsets. That can be used in federated learning context. 
        # All given datasets down below are available in ./dataset/download.
        #
        # dataset = Dataset(
        #     [
        #         (Btumor4600().ID, Btumor4600(), []),    # id 0 ID DATA 
        #         (Btumor3000().ID, Btumor3000(), []),    # id 1 ID DATA 
        #         (Balzheimer5100().ID, Balzheimer5100(), []), # id 2 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), []), # id 3 ID DATA 
        #         (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]), # id 4 ID DATA (subsamples of total, not used in pre-training) 
        #         (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]), # id 5 ID DATA (subsamples of total, not used in pre-training)  
        #         (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6 OOD DATA (poisoned data) (not used in pre-training)
        #         (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]]) # id 7 OOD DATA (not used in pre-training) # Take some two subsets of complete dataset. # [250,750], [4000,4500]
        #     ],
        #     dataset_config=dataset_config,
        #     plot_config=plot_config
        # )
        #
        # Bind local model / client to dataset id / subsets.
        #
        # train_data, validation_data, test_data = dataset.get(index i)
        #
        
        # Train the model on training data with validation monitoring
        m.train(train_data, validation_data)     
        # Evaluate the model on held-out test data
        m.test(test_data) 
        
        # Display training/validation plots and confusion matrix for test set
        m.plot_all(test_data, xlabel="CNN Model", title="CNN Model")
        # Wait for user input before closing plots
        input('Press Enter to close plots and exit...')
        
# Class for running federated learning simulation with multiple clients
class FederatedSimulation():
    """
        Federated learning simulation with multiple local models and a global model.
        Each client trains locally on their data, then aggregates updates to global model.
        
        Configuration can be customized for different simulation scenarios.
        Feel free to modify parameters as needed for your experiments.
    """
    
    # ============ FEDERATED LEARNING CONFIGURATION ============
    federated_config = ConfigFederated(
        debug = True,                       # Enable verbose output for federated training
            
        # _______ FILE MANAGEMENT _______
        save = False,                       # Whether to save global model after simulation
        load_round = 0,                     # Round to resume from when loading pretrained model
        load_reg = True,                    # Whether to continue training from loaded model
        load = False,                       # Whether to load model from disk (False = start fresh)
        delete_on_load = False,             # Delete loaded model file after loading
        path = "./.env/.saved/",            # Directory path for saving/loading models
            
        # _______ SIMULATION PARAMETERS _______
        rounds = 25,                        # Total number of federated training rounds
        ood_round = 26,                     # Round number when OOD detection starts (>rounds = never)
        clients = 5,                        # Total number of clients (global model + local models)
        participants = 4,                   # Number of clients participating per round (random selection)
        host_id=0,                          # ID of the global model server (typically 0)
        client_to_dataset=[[0,1,2,3],[0],[1],[2],[3]]   # Which datasets each client uses for training
    )
    
    # ============ OUT-OF-DISTRIBUTION DETECTION CONFIGURATION ============
    ood_config = ConfigOod(
        debug = True,                       # Enable OOD debug output
        hdc_debug = False,                  # Enable hyperdimensional computing debug output
            
        # _______ OOD PARAMETERS _______
        enabled = False,                    # Whether OOD detection is enabled
        hyper_size=int(1e4),                # Size of hyperdimensional projection (10,000 dimensions)
        
        id_client = [1,2,3,4],              # Client IDs with in-distribution data (normal clients)
        ood_client = [5,6],                 # Dataset indices with out-of-distribution data (for testing OOD)
        
        ood_protection = True,              # Whether to exclude OOD models from aggregation
        ood_protection_thres = 0.7          # Similarity threshold; below = OOD (exclude from average)
    )
    
    # ============ NEURAL NETWORK MODEL CONFIGURATION ============
    model_config = ConfigModel(
        debug = True,                       # Enable model training debug output
            
        # _______ TRAINING PARAMETERS _______
        epochs = 1,                         # Epochs per round (1 is recommended for federated)
        activation = 'relu',                # Activation function for hidden layers
        activation_out = 'softmax',         # Activation function for output (multi-class)
        optimizer = 'adam',                 # Optimization algorithm
        loss = 'categorical_crossentropy'   # Loss function for classification
    )
    
    # ============ DATASET CONFIGURATION ============
    dataset_config = ConfigDataset(
        debug = False,                      # Disable verbose dataset output
            
        # _______ DATASET PARAMETERS _______
        batch_size = 64,                    # Batch size for training iterations
        image_size = 256,                   # Resize images to 256x256 pixels
        input_shape = (256,256,1),          # Grayscale image input shape
        split = 0.25,                       # 25% for test/validation, 75% for training
        number_of_classes = 2               # Binary classification (2 output classes)
    )
    
    # ============ PLOTTING CONFIGURATION ============
    plot_config = ConfigPlot(
        plot = False,                       # Disable plots during federated training
            
        # _______ PLOT PARAMETERS _______
        path = './.env/plot',               # Directory for saving plot images
        img_per_class = 10                  # Sample images per class to display
    )
    
    def run(self):
        # ============ INITIALIZE MODEL ============
        m = Model(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        # Create dataset list: each entry is (dataset_id, dataset_object, [index_ranges])
        # Index ranges allow using subsets of data; empty [] = use all data
        dataset = Dataset(
            [
                (Btumor4600().ID, Btumor4600(), []),                                        # id 0: Full brain tumor dataset (4600 samples)
                (Btumor3000().ID, Btumor3000(), []),                                        # id 1: Brain tumor dataset (3000 samples)
                (Balzheimer5100().ID, Balzheimer5100(), []),                                # id 2: Alzheimer's dataset (5100 samples)
                (Lpneumonia5200().ID, Lpneumonia5200(), []),                                # id 3: Pneumonia dataset (5200 samples)
                (Lpneumonia5200().ID, Lpneumonia5200(), [[300,500],[3600,3800]]),          # id 4: Subset of pneumonia (not used in pre-training)
                (Btumor4600().ID, Btumor4600(), [[300,500],[3700,3900]]),                  # id 5: Subset of brain tumor (not used in pre-training)
                (Balzheimer5100_poisoned().ID, Balzheimer5100_poisoned(), [[1000,1700],[4000,4700]]), # id 6: OOD data (poisoned Alzheimer's)
                (Afaces16000().ID, Afaces16000(), [[1,700],[2501,3200]])                   # id 7: OOD data (animal faces - different domain)
            ],
            dataset_config=self.dataset_config,
            plot_config=self.plot_config
        )
        
        # ============ INITIALIZE FEDERATED LEARNING ============
        federated = Federated(
            dataset=dataset,                        # Dataset object for all clients
            model=m,                                # Template model to clone for each client
            federated_config=self.federated_config, # Federated training parameters
            ood_config=self.ood_config,             # OOD detection parameters
            dataset_config=self.dataset_config,     # Dataset configuration
            plot_config=self.plot_config            # Plotting configuration
        )
        
        # Run federated learning simulation and return results
        return federated.run()
    
# Entry point for the script
if __name__ == "__main__": 
    # Run a single model simulation (non-federated)
    #-------------------------
    sim1 = ModelSimulation()
    sim1.run()
    
    # Uncomment below to run federated learning simulation instead:
    # sim_federated = FederatedSimulation()
    # sim_federated.run()