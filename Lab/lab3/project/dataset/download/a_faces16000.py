# Import the parent class ImportKaggle for handling Kaggle dataset downloads
from dataset.download.utils.import_kaggle import ImportKaggle

class Afaces16000(ImportKaggle):
    # Class-level variable to store the dataset path (initialized as empty string)
    PATH = ""
    # Class-level variable to store the static path to data folders (initialized as empty string)
    STATIC_PATH = ""
    # Class-level variable to store the dataset identifier (initialized as empty string)
    ID = ""
    
    def __init__(self) -> None:
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # Kaggle dataset path: https://www.kaggle.com/datasets/andrewmvd/animal-faces
        # This identifies which dataset to download from Kaggle
        KAGGLE_PATH = "andrewmvd/animal-faces"
        
        # Path within the downloaded dataset folder to access the training images
        # Specifies the relative path to the folders containing class subfolders
        STATIC_PATH = '/afhq/train'  
        # List of subfolder names within the STATIC_PATH that contain images for different classes
        # These are the folder names in the Kaggle dataset structure
        paths = ['/cat', '/dog']
        # List of corresponding labels for each path folder (same length as paths)
        # Maps folder names to human-readable class labels (e.g., 'Healthy' or 'Sick')
        labels = ['Healthy', 'Sick']            
           
        # Set the dataset identifier using the current module name (__name__)
        # This is used for debugging and logging purposes to identify this dataset
        ID = __name__              
        
        # Call the parent class constructor with all required parameters
        # This initializes the ImportKaggle base class with the dataset configuration
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)