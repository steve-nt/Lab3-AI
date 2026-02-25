############################################################################################
#RECOMMENDED IS TO CHANGE CONFIGURATION IN SIMULATION INSTEAD OF DEFUALT VALUES GIVEN HERE.#
############################################################################################

# Import TensorFlow library for neural network operations
import tensorflow as tf

# Define a string prefix for configuration error messages
exception_msg = "Config.py incorrect parameterized: "

# Configuration class for federated learning simulation
class ConfigFederated():
    # Default values for federated learning parameters
    debug = False                          # Flag to enable verbose debugging output
    repeats = 1                            # Number of times to repeat the simulation
    
    # Save and load parameters for model persistence
    save = False                           # Whether to save models after training
    load_round = 40                        # Which round should federated sim continue from (from disk).
    load = False                           # Whether to load previously saved models from disk
    delete_on_load = False                 # Whether to delete loaded models from disk after loading
    path = "./.saved/"                     # Directory path where models are saved/loaded

    # Federated learning simulation parameters
    clients = 4                            # Clients (including server) in simulation.
    rounds = 5                             # Rounds of federated training.
    ood_round = 3                          # Round number when out-of-distribution detection starts
    participants = 3                       # Number of clients participating in each round (excludes global model)
    host_id = 0                            # ID of the global/server model (typically 0)
    client_to_dataset = [[0, 1, 2], [0], [1], [2]]   # Maps each client to datasets; length must equal clients
    
    def __init__(self, debug : bool, save : bool, load_round : int, load_reg : bool, load : bool, delete_on_load : bool, path : str, 
                 rounds : int, ood_round : int, clients : int, participants : int, host_id : int, client_to_dataset : list) -> None:
        """_summary_
        Federated learning configuration for simulation.

        Args:
            debug (bool): Enable detailed printouts in federated learning (more verbose output).
            repeats (int): Number of times to repeat the simulation.
            save (bool): Whether to save and load weights of trained clients after completed training.
            load_round (int): Which round of saved model should be loaded from disk.
            load (bool): If any model should be loaded from local disc. 
            delete_on_load (bool): Delete loaded model from disk after loading. 
            path (str): Path to folder for saving client weights.
            rounds (int): Rounds of training, iterations of client training.
            ood_round (int): Round to start out of distribution detection. 
            clients (int): Number of clients that will be created during sim (includes global model).
            participants (int): Number of participants that will train during each round (select n random from clients, excludes global). 
            host_id (int): ID/index of the global/server model (host). 
            client_to_dataset (list[list[int]]): Dataset allocation for each client. 

        """
        # Set the debug flag for verbose output
        self.debug = debug
        
        # Set save/load parameters
        self.save = save
        self.load = load
        self.load_reg = load_reg
        self.delete_on_load = delete_on_load
        self.path = path
        
        # Validate load_round: must be less than total rounds
        if(0 <= load_round < rounds):
            self.load_round = load_round
        elif (load):
            # If load is True, load_round must be valid
            raise Exception(exception_msg, "Must load round less than sim. rounds.")
        
        # Validate clients: must have at least 2 (one global model + one local model)
        if(clients >= 2):
            self.clients = clients
        else: 
            raise Exception(exception_msg,"Must have atleast two clients, one global model + one local model")
            
        # Validate rounds: must run for at least one round
        if(rounds >= 1):
            self.rounds = rounds
        else:
            raise Exception(exception_msg, "Must run for atleast one round")
        
        # Validate ood_round: must be greater than 0
        if(ood_round >= 1):
            self.ood_round = ood_round
        else:
            raise Exception(exception_msg, "ood rounds must be greater than 0")
        
        # Validate participants: must be less than total number of clients
        if(participants < clients):
            self.participants = participants
        else:
            raise Exception(exception_msg, "Number of participants must be less than clients")
        
        # Validate host_id: must be within valid client ID range
        if(0 <= host_id <= (clients-1)):
            self.host_id = host_id
        else:
            raise Exception(exception_msg, "Host id not valid ")
        
        # Validate client_to_dataset: length must match number of clients
        if(len(client_to_dataset) == clients): 
            self.client_to_dataset = client_to_dataset
        else:
            raise Exception(exception_msg, "Not all clients are assigned a dataset, length of ")

# Configuration class for Out-Of-Distribution (OOD) detection using Hyperdimensional Feature Fusion
class ConfigOod():
    # Debug flags for verbose output
    debug = True                           # Enable debug printouts
    hdc_debug = True                       # Enable Hyperdimensional Computing debug printouts
    enabled = False                        # Whether OOD detection is enabled in the simulation
    
    # Hyperdimensional computing parameters
    hyper_size = int(1e4)                  # Size of hyperdimensional projection space (typically 1000-10000)
    cosine_sum = tf.math.reduce_sum,      # Mathematical operation for cosine similarity computation
    
    # Client classification for OOD detection
    id_client = []                         # List of client IDs with in-distribution (normal) data
    ood_client = []                        # List of client IDs with out-of-distribution (abnormal) data
    
    # OOD protection/filtering parameters
    ood_protection = False                 # Whether to exclude models detected as OOD from aggregation
    ood_protection_thres = 0.2             # Threshold value: similarity score <= this value indicates OOD

    def __init__(self, debug : bool, hdc_debug : bool, enabled : bool, hyper_size : int, id_client : list, ood_client : list, 
                 ood_protection : bool, ood_protection_thres : float):
        """ Configuration for hyperdimensional features and OOD detection.

        Args:
            debug (bool): Enable verbose output and prompts.
            hdc_debug (bool): Enable detailed output in hyperdimensional module.
            enabled (bool): Whether to enable OOD detection module.
            hyper_size (int): Dimension of hyperdimensional projection space (higher = more precision). 
            id_client (list): List of client IDs with in-distribution data for model training.
            ood_client (list): List of client IDs with out-of-distribution data for OOD testing.
            ood_protection (bool): If True, exclude models with OOD scores below threshold from aggregation. 
            ood_protection_thres (float): Threshold value for OOD detection; similarity <= this is considered OOD (range: 0-1). 

        Raises:
            Exception: If hyper_size is not positive.
            Exception: If ood_protection_thres is not in valid range (0, 1).
        """
        # Set debug flags
        self.debug = debug
        self.hdc_debug = hdc_debug
        
        # Set OOD detection configuration
        self.enabled = enabled
        self.id_client = id_client
        self.ood_client = ood_client
        
        # Set OOD protection parameters
        self.ood_protection = ood_protection
        
        # Validate hyper_size: must be positive integer
        if(0 < hyper_size):
            self.hyper_size = hyper_size
        else:
            raise Exception(exception_msg, "Hyper size incorrect in hdff, must be >0")
        
        # Validate ood_protection_thres: must be between 0 and 1 (exclusive)
        if(0 < ood_protection_thres < 1):
            self.ood_protection_thres = ood_protection_thres
        else:
            raise Exception(exception_msg, "ood protection threshold is incorrect, must be 0<x<1")

    
# Configuration class for neural network model training parameters
class ConfigModel():
    # Debug and training parameters
    debug = True                           # Enable debug output during training
    
    # Neural network hyperparameters
    epochs = 15                            # Number of training epochs (complete passes through training data)
    activation = 'relu'                    # Activation function for hidden layers (Rectified Linear Unit)
    activation_out = 'softmax'             # Activation function for output layer (for multi-class classification)
    optimizer = 'adam'                     # Optimizer algorithm for gradient descent (Adaptive Moment Estimation)
    loss = 'categorical_crossentropy'      # Loss function for multi-class classification
    
    def __init__(self, debug, epochs, activation, activation_out, optimizer, loss) -> None:
        """
        Model configuration for simulation.

        Parameters
        ----------
            debug : bool
                Enable printouts/debug messages during model execution and training.
            epochs : int
                Number of complete passes through training data (training iterations).
            activation : str
                Activation function for hidden layers (tensorflow.keras compatible string). 
            activation_out : str
                Activation function for output layer (tensorflow.keras compatible string).
            optimizer : str
                Optimizer algorithm for weight updates (tensorflow.keras compatible string).
            loss : str 
                Loss function to minimize during training (tensorflow.keras compatible string).
        """
        # Set debug flag
        self.debug = debug
        
        # Validate epochs: must be at least 1
        if(self.epochs >= 1):
            self.epochs = epochs
        else:
            raise Exception(exception_msg, "Must atleast run for 1 epoch")
        
        # Set neural network layer parameters
        self.activation = activation
        self.activation_out = activation_out
        self.optimizer = optimizer
        self.loss = loss

# Configuration class for dataset handling and preprocessing
class ConfigDataset():
    # Debug flag for verbose output
    debug = True                           # Enable debug output during dataset operations
    
    # Dataset parameters
    batch_size = 64                        # Number of samples processed before weight update
    image_size = 256                       # Target size for resizing images (256x256 pixels)
    input_shape = (256,256,3)              # Target input shape for model (height, width, channels)
    
    # Dataset split parameters
    split = 0.25                           # Fraction of data used for test/validation (0.25 = 25%) 
    
    def __init__(self, debug, batch_size, image_size, input_shape, split, number_of_classes) -> None:
        """
        Dataset configuration for simulation.

        Parameters
        ----------
            debug : bool
                Enable printouts and debug messages during dataset operations and loading.
            batch_size : int
                Number of samples to process together in each training iteration. 
            image_size : int
                Target pixel size for image resizing (creates square images: image_size x image_size).
            input_shape : tuple(int,int,int)
                Target input shape for images and model input (height, width, channels).
            split : float
                Fraction of data for test and validation splits (between 0 and 1). 
            number_of_classes : int
                Total number of output classes for classification task.
        """
        
        # Set debug flag
        self.debug = debug
        self.batch_size = batch_size
        
        # Validate image_size matches input_shape dimensions
        if((image_size == input_shape[0]) and (image_size == input_shape[1])):
            self.image_size = image_size
            self.input_shape = input_shape
        else:
            raise Exception(exception_msg, "Dataset dimensions incorrect")
        
        # Validate split: must be a valid fraction between 0 and 1
        if(0 < split < 1):
            self.split = split
        else:
            raise Exception(exception_msg, "split is percentage in float")
        
        # Validate number_of_classes: must be positive
        if(number_of_classes > 0):
            self.number_of_classes = number_of_classes
        else:
            raise Exception(exception_msg, "number of classes must be greater than 0")
            
# Configuration class for plotting and visualization
class ConfigPlot():
    # Plotting parameters
    plot = True                            # Whether to enable/display plots during execution
    path = './.env/plot'                   # Directory path for saving plot images
    img_per_class = 10                     # Number of sample images to display per dataset class 
    
    def __init__(self, plot : bool, path : str, img_per_class : int) -> None:
        """
        Plot configuration for simulation.
        
        Parameters
        ----------
            plot : bool
                If plotting is enabled during simulation execution.
            path : str
                Directory path for saving generated plot images/figures. 
            img_per_class : int
                Number of sample images per class to show in dataset visualization plots.
        """
        # Set plotting parameters
        self.plot = plot
        self.path = path
        self.img_per_class = img_per_class