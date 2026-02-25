# Import the parent class ImportKaggle for handling Kaggle dataset downloads
from dataset.download.utils.import_kaggle import ImportKaggle

class Btumor4600(ImportKaggle):  
    # Class-level variable to store Kaggle dataset path (initialized as empty string)
    KAGGLE_PATH = ""
    # Class-level variable to store the static path to data folders (initialized as empty string)
    STATIC_PATH = ""
    # Class-level variable to store the dataset identifier (initialized as empty string)
    ID = ""
    
    def __init__(self) -> None:  
        """
            Structure for downloading dataset. Inheritance from utils. Kaggle dataset -> ImportKaggle.py . 
        """
        
        # Kaggle dataset path: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
        # This identifies which dataset to download from Kaggle
        KAGGLE_PATH = "preetviradiya/brian-tumor-dataset"
        
        # Path within the downloaded dataset folder to access the training images
        # Specifies the relative path to the folders containing class subfolders
        STATIC_PATH = '/Brain Tumor Data Set/Brain Tumor Data Set'  
        # List of subfolder names within the STATIC_PATH that contain images for different classes
        # Two categories: Brain Tumor presence or Healthy (no tumor)
        paths = ['/Brain Tumor', '/Healthy']
        # List of corresponding labels for each path folder (same length as paths)
        # Maps folder names to binary health status labels: tumor present or not
        labels = ['Sick', 'Healthy']             
           
        # Set the dataset identifier using the current module name (__name__)
        # This is used for debugging and logging purposes to identify this dataset
        ID = __name__              
        
        # Call the parent class constructor with all required parameters
        # This initializes the ImportKaggle base class with the dataset configuration
        super().__init__(ID=ID, KAGGLE_PATH=KAGGLE_PATH, STATIC_PATH=STATIC_PATH, paths=paths, labels=labels)
    




