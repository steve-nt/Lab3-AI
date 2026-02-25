# Import Dataset configuration class
from config import ConfigDataset, ConfigPlot
# Import dataset download classes
from dataset.download.b_alzheimer5100 import Balzheimer5100
from dataset.download.b_tumor3000 import Btumor3000
from dataset.download.b_tumor4600 import Btumor4600
# Import dataset generator for creating train/val/test splits
from dataset.generator import DatasetGenerator

# Class for managing and organizing datasets from multiple sources
class Dataset:
    # Class variables for storing datasets
    datasets = []              # List of loaded datasets
    dataset_config = None      # Reference to dataset configuration
    plot_config = None         # Reference to plotting configuration
    
    def __init__(self, datasets : list, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Packages multiple datasets and prepares them for training.
        
        Parameters
        ----------
            datasets : list[(str(ID), dataset_object, [int_ranges])]
                List of tuples: each contains dataset ID, dataset object, and optional index ranges.
            dataset_config : ConfigDataset 
                Dataset configuration (batch size, image size, preprocessing).
            plot_config : ConfigPlot 
                Plotting configuration for visualization.
        """
        # Store the datasets list
        self.datasets = datasets
        # Store configuration references
        self.dataset_config = dataset_config
        self.plot_config = plot_config
    
    def split_indicies(self, result_paths, result_labels, indicies):
        """
        Extract data within specified index ranges.
        
        Parameters
        ----------
            result_paths : list
                Paths to all images.
            result_labels : list
                Labels corresponding to image paths.
            indicies : list
                List of [start, end] ranges to extract.
        
        Returns
        -------
            tuple : (extracted_paths, extracted_labels) within specified ranges.
        """
        # Validate index range is valid: start >= 0 and start < end
        if(0 <= indicies[0] < indicies[1]):
            # Return slice of data between start and end indices
            return result_paths[indicies[0]:indicies[1]], result_labels[indicies[0]:indicies[1]]
        else:
            raise Exception("Range in dataset is incorrect")
    
    def mergeAll(self):
        """
        Merge all datasets into a single combined dataset.
        Returns train, validation, and test data iterators.
        
        Returns
        -------
        tuple : (train_data, validation_data, test_data) as TensorFlow iterators
        
        Raises
        ------
        Exception : If no datasets are loaded.
        """
        
        # Validate that datasets list is not empty
        if len(self.datasets) == 0:
            raise Exception("No datasets loaded in DatasetGenerator.")
        # Validate that each dataset entry has 3 components (id, object, indices)
        for x in self.datasets:
            if len(x) != 3:
                raise Exception("Must specify name (string), dataset (ImportKaggle), indicies of dataset ([int,int])")
        
        # Initialize lists to store paths and labels from all datasets
        result_paths = []
        result_labels = []
        result_indicies = []
        
        # Debug: print all datasets being merged
        print(self.datasets)
        # Iterate through each dataset and collect paths/labels
        for x in self.datasets:
            print(x)
            # Collect all image paths from current dataset
            for y in x[1].paths:
                result_paths.append(y)
            # Collect all labels from current dataset
            for y in x[1].labels:
                result_labels.append(y)
            # Collect index ranges from current dataset
            result_indicies = result_indicies + x[2]
                
        # Create dataset generator for splitting into train/val/test
        dataset = DatasetGenerator()
        
        # Generate and return train, validation, test data iterators
        return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, None)
        
    def get(self, *args):
        """
        Get train/val/test data for specific dataset(s) by index or list of indices.
        
        Parameters
        ----------
            *args : int or list[int]
                Single index (int) or list of indices to retrieve.
        
        Returns
        -------
        tuple : (train_data, validation_data, test_data) iterators for specified dataset(s).
        
        Raises
        ------
        Exception : If index is out of range.
        """
        # Initialize lists for collecting data
        result_paths = []
        result_labels = []
        result_indicies = []
        
        # Handle single dataset index
        if len(args) == 1 and isinstance(args[0], int):
            i = args[0]
            # Validate index is in range
            if(0 <= i <= (len(self.datasets) - 1)):
                # Get paths, labels, and indices for dataset at index i
                result_paths = self.datasets[i][1].paths
                result_labels = self.datasets[i][1].labels
                result_indicies = result_indicies + self.datasets[i][2]
                
                # Create dataset generator
                dataset = DatasetGenerator()
                
                # Generate and return data with dataset-specific preprocessing
                return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
            else:
                raise Exception("Index out of range.")
        
        # Handle list of dataset indices
        if len(args) == 1 and isinstance(args[0], list):
            # Iterate through each index in the list
            for i in args[0]:
                # Validate each index is in range
                if(0 <= i <= (len(self.datasets) - 1)):
                    # Combine paths, labels, and indices from multiple datasets
                    result_paths = result_paths + self.datasets[i][1].paths
                    result_labels = result_labels + self.datasets[i][1].labels
                    result_indicies = result_indicies + self.datasets[i][2]
                else:
                    raise Exception("Index out of range.")
                
            # Create dataset generator
            dataset = DatasetGenerator()
            # Generate and return merged data from multiple datasets
            return dataset.generate(result_paths, result_labels, result_indicies, self.dataset_config, self.plot_config, lambda image: self.datasets[i][1].pre_processing(image))
        
    def print(self):
        """
        Print information about all loaded datasets.
        """
        # Iterate and print each dataset entry
        for x in (self.datasets):
            print(x)
  


        