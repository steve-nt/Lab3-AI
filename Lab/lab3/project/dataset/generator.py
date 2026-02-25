# Import NumPy for numerical operations
import numpy as np
# Import OpenCV for image processing
import cv2

# Import TensorFlow for deep learning
import tensorflow as tf
# Import Keras image preprocessing utilities
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import scikit-learn for train/test splitting with stratification
from sklearn.model_selection import train_test_split

# Import configuration classes
from config import ConfigDataset, ConfigPlot
# Import dataset visualization utilities
from dataset.math.plot import DatasetPlot
# Import dataframe generation utilities
from dataset.gen.dataframe import DataframeGenerator

# Class for generating TensorFlow data iterators from image directories
class DatasetGenerator:
    def generate(self, paths, labels, indicies, dataset_config, plot_config, image_processing):
        """
        Generate train, validation, and test data iterators from image paths.
        Applies image preprocessing, splitting, and augmentation.
    
        Args:
            paths (list): List of paths to image directories.
            labels (list): List of labels corresponding to each path.
            indicies (list): List of [start, end] ranges to subset data (empty list = use all).
            dataset_config (ConfigDataset): Dataset configuration (batch size, image size, split ratio).
            plot_config (ConfigPlot): Plotting configuration.
            image_processing (func): Optional custom image preprocessing function.

        Returns:
            tuple: (train_data, validation_data, test_data) TensorFlow iterators.
        """
        
        # Generate dataframe with file paths and labels
        df = DataframeGenerator.generate(paths, labels, indicies)
        
        # Print dataframe info if debug is enabled
        if(dataset_config.debug):
            print("\nDataframe:")
            print(df.head())  # Show first 5 rows
            print("")
            print(df.tail())  # Show last 5 rows
            print("")
            
        # Split data into train (75%) and test (25%) sets with stratification by label
        train_df, test_df = train_test_split(df, 
                                             test_size=dataset_config.split,  # 25% test size
                                             random_state=42,                  # For reproducibility
                                             stratify=df.labels)               # Keep class distribution
        # Further split training data into train (90%) and validation (10%) sets
        train_df, val_df = train_test_split(train_df, 
                                            test_size=(dataset_config.split-0.1),  # 10% validation
                                            random_state=42,
                                            stratify=train_df.labels)
        
        # Print split sizes if debug is enabled
        if(dataset_config.debug):
            print("\nDataframe shapes (split): \n")
            print("Training dataset   : ", train_df.shape)
            print("Test dataset       : ",test_df.shape)
            print("Validation dataset : ", val_df.shape)
            print("")
        
        # Configure image generator with preprocessing and augmentation
        # Fix pre-processing for single dataset
        if(image_processing != None):
            # Use custom preprocessing if provided
            image_gen = ImageDataGenerator(
                rescale=1./255,                                                      # Normalize pixel values to [0, 1]
                preprocessing_function=lambda image: self.default_processing(image)  # Apply default preprocessing
        )
        else:
            # Use default preprocessing if no custom function provided
            image_gen = ImageDataGenerator(
                rescale=1./255,                                                      # Normalize pixel values to [0, 1]
                preprocessing_function=lambda image: self.default_processing(image)  # Apply default preprocessing
        )
        
        # CHANGED COLOR MODE TO GRAYSCALE for medical imaging
        # Create training data iterator from dataframe
        train_data = image_gen.flow_from_dataframe(
            dataframe=train_df,                                              # Training dataframe
            x_col="filepaths",                                              # Column with image paths
            y_col="labels",                                                 # Column with labels
            target_size=(dataset_config.image_size, dataset_config.image_size),  # Resize images
            color_mode='grayscale',                                         # Convert to grayscale
            class_mode="categorical",                                       # One-hot encoded labels
            batch_size=dataset_config.batch_size,                          # Batch size for training
            shuffle=True)                                                   # Shuffle data
        
        # Create validation data iterator from dataframe
        validation_data = image_gen.flow_from_dataframe(
            dataframe=val_df,
            x_col="filepaths",
            y_col="labels",
            target_size=(dataset_config.image_size, dataset_config.image_size),
            color_mode='grayscale',
            class_mode="categorical",
            batch_size=dataset_config.batch_size,
            shuffle=False)                                                  # Don't shuffle validation
        
        # Create test data iterator from dataframe
        test_data = image_gen.flow_from_dataframe(
            dataframe=test_df,
            x_col="filepaths",
            y_col="labels",
            target_size=(dataset_config.image_size, dataset_config.image_size),
            color_mode='grayscale',
            class_mode="categorical",
            batch_size=dataset_config.batch_size,
            shuffle=False)                                                  # Don't shuffle test
        
        # Plot dataset class distribution if enabled
        if(plot_config.plot and dataset_config.debug):
            plot = DatasetPlot()
            plot.plot_classes(train_data, train_df, plot_config)
        
        # Return train, validation, and test iterators
        return train_data, validation_data, test_data

    def default_processing(self, image):
        """
        Apply default image preprocessing pipeline for medical images.
        Includes: grayscale conversion, contrast enhancement, sharpening, clipping.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Preprocessed image with channel dimension.
        """
        # Convert RGB to grayscale if image is color (3 channels)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Remove unnecessary dimension if single-channel (H, W, 1) -> (H, W)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]

        # Apply contrast enhancement using weighted sum: output = image*1.5 - image*0.5
        image = cv2.addWeighted(image, 1.5, image, -0.5, 0)

        # Apply sharpening kernel to enhance edges and details
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

        # Scale and clip pixel values to valid range [0, 255]
        image = np.clip(image * 1.25, 0, 255)

        # Restore channel dimension: (H, W) -> (H, W, 1) for model input
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # Return preprocessed image
        return image