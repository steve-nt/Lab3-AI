# Import matplotlib pyplot for creating and displaying plots
import matplotlib.pyplot as plt
# Import numpy for numerical operations and array manipulation
import numpy as np
# Import PCA (Principal Component Analysis) from scikit-learn for dimensionality reduction
from sklearn.decomposition import PCA
# Import StandardScaler from scikit-learn for normalizing feature data
from sklearn.preprocessing import StandardScaler

class DatasetPlot:
    # Class for visualizing dataset images from different classes
    def plot_classes(self, dataset, dataset_df, plot_config):
        # Get list of class labels from the dataset's class_indices dictionary
        # class_indices maps class names to their indices (e.g., {'cat': 0, 'dog': 1})
        class_labels = list(dataset.class_indices.keys())

        # Create a new figure with size 15x10 inches for displaying multiple images
        plt.figure(figsize=(15, 10))
        # Iterate through each class label in the dataset
        for label in class_labels:
            # Filter the dataframe to get all rows where the 'labels' column matches the current class
            indices = dataset_df[dataset_df['labels'] == label].index
            
            # Randomly sample up to plot_config.img_per_class indices from the current class
            # The min() ensures we don't try to sample more images than available
            # replace=False means each index can only be selected once
            indices = np.random.choice(indices, min(plot_config.img_per_class, len(indices)), replace=False)
            
            # Iterate through the sampled indices for the current class
            for i, idx in enumerate(indices):
                # Calculate the subplot position (row, column) in the grid
                # Grid layout: rows = number of classes, columns = plot_config.img_per_class
                # Formula: len(class_labels) rows × plot_config.img_per_class columns
                plt.subplot(len(class_labels), plot_config.img_per_class, len(class_labels)*i + class_labels.index(label) + 1)
                # Read and display the image from the filepath stored in the dataframe
                # The image file path is stored in the 'filepaths' column at index idx
                plt.imshow(plt.imread(dataset_df.loc[idx, 'filepaths']))  
                # Set the subplot title to the class label name
                plt.title(label)
                # Turn off axis labels and ticks for cleaner visualization
                plt.axis('off')
        # Display the complete figure with all class images
        plt.show()

    