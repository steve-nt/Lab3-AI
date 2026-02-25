# Import NumPy for numerical operations
import numpy as np
# Import Pandas for dataframe operations
import pandas as pd
# Import os for file system operations
import os 

# Class for generating pandas DataFrames with image paths and labels
class DataframeGenerator:
    def generate(paths, labels, indicies) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with image file paths and their corresponding labels.
        Optionally filters data to specific index ranges.
        
        Parameters
        ----------
        paths : list[list[str]]
            List of lists containing directory paths where images are stored.
        labels : list[str]
            List of labels corresponding to each directory in paths (parallel list).
        indicies : list[[int, int]]
            List of [start, end] ranges to filter data. Empty list [] uses all data.
        
        Returns
        ----------
        pd.DataFrame
            DataFrame with columns 'filepaths' and 'labels' for all images.
        """
        
        # Initialize lists to store all file paths and labels
        result_paths = []
        result_labels = []
        
        # Iterate through each directory and its corresponding label
        for i, dir_list in enumerate(paths):
            # Iterate through each subdirectory in the current directory group
            for j in dir_list:
                # List all files in the directory
                list_f = os.listdir(j)
                # Process each file found in the directory
                for name in list_f:
                    # Create full file path by joining directory and filename
                    fpath = os.path.join(j, name)
                    # Add the full file path to results
                    result_paths.append(fpath)
                    # Add the corresponding label to results
                    result_labels.append(labels[i])

        # Initialize temporary lists for filtered data
        temp_paths = []
        temp_labels = []
        # Apply index range filtering if provided (not empty)
        for x in indicies:
            # Validate index range is valid: start >= 0, end valid, start < end
            if(0 <= x[0] < x[1] <= len(result_paths) and x != []):
                # Extract paths within the specified range
                for y in result_paths[x[0]:x[1]]:
                    temp_paths.append(y)
                # Extract labels within the specified range
                for y in result_labels[x[0]:x[1]]:
                    temp_labels.append(y)
            else:
                raise Exception("Indicies for dataset is incorrect.")
        
        # Use filtered data if ranges were provided, otherwise use all data
        if(temp_paths != [] and temp_labels != []):
            result_paths = temp_paths
            result_labels = temp_labels
        
        # Convert paths list to pandas Series with name 'filepaths'
        result_paths = pd.Series(result_paths, name="filepaths")
        # Convert labels list to pandas Series with name 'labels'
        result_labels = pd.Series(result_labels, name="labels")
        
        # Concatenate paths and labels Series into a single DataFrame
        result_data = pd.concat([result_paths, result_labels], axis=1)
        
        # Return the resulting DataFrame
        return pd.DataFrame(result_data)