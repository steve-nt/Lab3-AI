# Import TensorFlow for neural network operations
import tensorflow as tf
# Import Keras Sequential API for building neural networks layer by layer
from tensorflow.keras.models import Sequential
# Import layer types for building the neural network
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

# Import configuration classes
from config import ConfigDataset, ConfigModel, ConfigPlot
# Import plotting utilities for visualizing training results
from model.math.plot import ModelPlot

# Class for creating and training a single neural network model
class Model:
    def __init__(self, model_config : ConfigModel, dataset_config : ConfigDataset, plot_config : ConfigPlot) -> None:
        """
        Single model structure for training and image classification. 
        Creates a CNN model, trains it, and evaluates it on test data.

        Parameters
        ----------
            model_config : ConfigModel
                Model hyperparameters (epochs, activation functions, optimizer, loss).
            dataset_config : ConfigDataset
                Dataset parameters (input shape, batch size, preprocessing).
            plot_config : ConfigPlot
                Plotting configuration (whether to display plots, where to save them).
        """
        # Create plot object for visualization
        self.plot = ModelPlot()
        # Initialize model reference (will be populated below)
        self.model = None
        
        # Training history tracking - number of epochs completed so far
        self.epochs = 0         # Useful for federated learning to track local model contributions
        
        # Training history metrics - track accuracy and loss across epochs
        self.acc = None         # Training accuracy history
        self.val_acc = None     # Validation accuracy history
        self.loss = None        # Training loss history
        self.val_loss = None    # Validation loss history

        # Test metrics - track accuracy and loss on test set
        self.test_accuracy = [] # List of test accuracy values from each test phase
        self.test_loss = []     # List of test loss values from each test phase
        
        # Store configuration objects for later use
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.plot_config = plot_config
        
        # ------------------------------- BUILD MODEL ARCHITECTURE -------------------------------
        
        # Create a sequential model (layers stacked one after another)
        model = Sequential()
        
        # Add input layer specifying input shape
        model.add(InputLayer(input_shape=self.dataset_config.input_shape, batch_size=self.dataset_config.batch_size))
        # Add first convolutional layer: 16 filters, 16x16 kernel size
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        # Add max pooling layer: reduces spatial dimensions (default 2x2)
        model.add(MaxPooling2D())
            
        # Add second convolutional layer
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        # Add max pooling layer
        model.add(MaxPooling2D())
        
        # Add third convolutional layer
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        # Add max pooling layer
        model.add(MaxPooling2D())
            
        # Add fourth convolutional layer
        model.add(Conv2D(16, 16, activation=self.model_config.activation))
        # Add max pooling layer
        model.add(MaxPooling2D())
            
        # Flatten the 4D feature maps into 1D vector for dense layers
        model.add(Flatten())
        # Add dense (fully connected) layer with 32 neurons
        model.add(Dense(32, activation=self.model_config.activation))
        # Add output layer with number of neurons equal to number of classes
        model.add(Dense(2, activation=self.model_config.activation_out))
        
        # --------- COMPILE MODEL WITH OPTIMIZER AND LOSS FUNCTION ---------
        
        # Compile model: set optimizer, loss function, and evaluation metrics
        model.compile(optimizer=self.model_config.optimizer, loss=self.model_config.loss, metrics=['accuracy'])
        
        # Display model architecture summary if debug mode is enabled
        if(self.model_config.debug):
            model.summary()
        
        # Store the created model
        self.model = model

    def train(self, train_data, val_data):
        """
        Train the neural network model on training data and validate on validation data.

        Parameters
        ----------
            train_data : tensorflow dataset
                Training dataset iterator with images and labels.
            val_data : tensorflow dataset
                Validation dataset iterator for monitoring training progress.
        """
        
        # Fit the model: train for specified number of epochs starting from current epoch count
        history = self.model.fit(train_data,
                    epochs=int(self.epochs + self.model_config.epochs),  # Total epochs to run
                    validation_data=val_data,                             # Validation set for monitoring
                    initial_epoch=self.epochs)                            # Resume from current epoch
        
        # Store training history for first training session
        if(self.epochs == 0):
            # Extract accuracy and loss from history for first time
            self.acc = history.history['accuracy']
            self.val_acc = history.history['val_accuracy']

            self.loss = history.history['loss']
            self.val_loss = history.history['val_loss']
        else:
            # Append to existing history if training continues (federated learning scenario)
            self.acc += history.history['accuracy']
            self.val_acc += history.history['val_accuracy']

            self.loss += history.history['loss']
            self.val_loss += history.history['val_loss']
        
        # Update total epochs count    
        self.epochs = int(self.epochs + self.model_config.epochs)
        
    def test(self, test_data):
        """
        Evaluate model performance on test dataset.
        
        Parameters
        ----------
            test_data : tensorflow dataset
                Test dataset iterator with images and labels.
        """
        # Evaluate model on test data: returns [loss, accuracy]
        loss, accuracy = self.model.evaluate(test_data)
        
        # Store test results for later visualization
        self.test_accuracy.append(accuracy)
        self.test_loss.append(loss)

        # Print test results if debug mode is enabled
        if(self.model_config.debug):
            print("Test loss= ", loss)
            print("Test accuracy=", accuracy)
            print()
            
    def plot_all(self, test_data, xlabel : str, title : str):
        """
        Plot training/validation history and confusion matrix on test data.
        
        Parameters
        ----------
            test_data : tensorflow dataset
                Test dataset for generating confusion matrix.
            xlabel : str
                Label for x-axis of plots.
            title : str
                Title for plot window.
        """
        # Plot accuracy and loss curves if training history exists
        if(self.acc != None):
            self.plot.accuracy(self.acc, self.val_acc, self.loss, self.val_loss, xlabel, title)
        # Plot confusion matrix showing prediction accuracy per class
        self.plot.confusion_matrix(self.model, test_data, title)
        
    def plot_test(self, xlabel : str, title : str):
        """
        Plot test accuracy and loss across multiple test phases.
        
        Parameters
        ----------
            xlabel : str
                Label for x-axis of plots.
            title : str
                Title for plot window.
        """
        # Display test accuracy and loss plots
        self.plot.test_accuracy(self.test_accuracy, self.test_loss, xlabel, title)
    