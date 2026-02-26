# Import cv2 (OpenCV) library for image processing operations
import cv2
# Import kagglehub for downloading datasets from Kaggle
import kagglehub
# Import numpy for numerical array operations and image manipulation
import numpy as np

class ImportKaggle():
    # Class-level variable to store paths to dataset folders (initialized as None)
    paths = None
    # Class-level variable to store label mappings for dataset classes (initialized as None)
    labels = None
    
    def __init__(self, ID, KAGGLE_PATH, STATIC_PATH, paths, labels) -> None:
        """
            Import utility for importing datasets from kaggle.
        Args:
            ID (str): Identifier for dataset (can be any)
            KAGGLE_PATH (str): Path to dataset on kaggle site, (last in url).
            STATIC_PATH (str): Path to subfolder's with images on local machine, (folder strucutre may be different from datasets).
            paths (str): Name of subfolders (as path).
            labels (str): Target labels, what new labels should be for all images in subfolders, same length as paths, position in labels correspond to postion in paths. 
        """
        # Store the dataset identifier as an instance variable for later debugging/reference
        self.ID = ID    # Identifier in debug for name of dataset. 
        
        # Print a message indicating that dataset download is starting
        print("Downloading dataset..." + self.ID + '\n')
        
        # Download the dataset from Kaggle using the kagglehub API
        # Returns the local path where the dataset was downloaded
        PATH = kagglehub.dataset_download(KAGGLE_PATH)
        # Print the local path where dataset files have been downloaded
        print("Path to dataset files:", PATH)
        # Print blank line for formatting
        print()
        
        # Append the STATIC_PATH to the base PATH to get the full path to image folders
        # This is necessary because the dataset structure may differ from Kaggle structure
        PATH = PATH + STATIC_PATH  # Must specifiy path to where path of subfolders of pictures exist. 
        
        # Initialize an empty list to store the full paths to each subfolder
        paths_result = []          # Construct seperate paths to each subfolder with images. 
        # Iterate through each provided path
        for x in paths:
            # Append the full path (base PATH + subfolder name) to the paths_result list
            paths_result.append([PATH + x])
        
        # Store the complete list of paths to image subfolders as an instance variable
        self.paths = paths_result
        # Store the corresponding labels for each path as an instance variable
        self.labels = labels
        
    def pre_processing(image):
        """
            Basic pre processing function. Can be overrided in inheritance. Currying in dataset generator. 
        Args:
            image (img): Image as input when currying.

        Returns:
            img: processed image. 
        """
        # Apply weighted addition to enhance image contrast
        # Weights: 1.5 on original + (-0.5) on original = result is image multiplied by 1.0 with contrast boost
        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

        # Create a sharpening kernel (high-pass filter)
        # [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] enhances edges and details
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Apply the sharpening kernel to the image
        # -1 means the kernel will use the same data type as the source image
        image = cv2.filter2D(image, -1, kernel)

        # Extract the red channel (index 2 in BGR format) from the image
        value = image[:, :, 2]
        # Increase the intensity of the red channel by 1.25x and clip values to valid range [0, 255]
        value = np.clip(value * 1.25, 0, 255)

        # Assign the processed red channel back to the image
        image[:, :, 2] = value

        # Return the processed image with contrast, sharpness, and color enhancements
        return image