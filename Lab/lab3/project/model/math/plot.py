# Import TensorFlow for neural network operations and evaluation
import tensorflow as tf
# Import numpy for numerical operations and array manipulation
import numpy as np

# Import seaborn for creating enhanced heatmaps (used for confusion matrix visualization)
import seaborn as sns

# Import matplotlib pyplot for creating and displaying plots
import matplotlib.pyplot as plt
# Import matplotlib ticker for customizing axis labels and formatting
import matplotlib.ticker as mticker

class ModelPlot:
    # Method to plot training/validation accuracy and loss curves
    def accuracy(self, train_accuracy, val_accuracy, train_loss, val_loss, xlabel : str,  title : str):
        # Create a new figure with custom title and size (14x5 inches)
        # num=title uses the title as the figure identifier for matplotlib
        plt.figure(num=title, figsize=(14, 5))

        # Plot training and validation accuracy in the first subplot (position 2,3,4)
        plt.subplot(2, 3, 4)
        # Set the x-axis to display only integer values (one tick per epoch)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot training accuracy line in blue color
        plt.plot(train_accuracy, label='Training Accuracy', color='blue')
        # Set the x-axis again to ensure integer values
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot validation accuracy line in orange color
        plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
        # Set the x-axis label with the provided xlabel parameter
        plt.xlabel(xlabel)
        # Set the y-axis label to 'Accuracy'
        plt.ylabel('Accuracy')
        # Set the subplot title combining provided title with descriptive text
        plt.title('Training and Validation Accuracy ' + title)
        # Add a legend to identify the training and validation lines
        plt.legend()

        # Plot training and validation loss in the second subplot (position 2,3,5)
        plt.subplot(2, 3, 5)
        # Set the x-axis to display only integer values (one tick per epoch)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot training loss line in blue color
        plt.plot(train_loss, label='Training Loss', color='blue')
        # Set the x-axis again to ensure integer values
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot validation loss line in orange color
        plt.plot(val_loss, label='Validation Loss', color='orange')
        # Set the x-axis label with the provided xlabel parameter
        plt.xlabel(xlabel)
        # Set the y-axis label to 'Loss'
        plt.ylabel('Loss')
        # Set the subplot title combining provided title with descriptive text
        plt.title('Training and Validation Loss ' + title)
        # Add a legend to identify the training and validation lines
        plt.legend()

        # Adjust layout to prevent overlapping of subplots and labels
        plt.tight_layout()
        # Enable interactive mode: allows plots to be displayed without blocking execution
        # Multiple plots can be launched and viewed without blocking the program
        plt.ion()  

        # Display the complete figure with all subplots
        plt.show()
        
    def test_accuracy(self, accuracy, loss, xlabel : str, title : str):
        # Create a new figure with custom title and size (14x5 inches) for test results
        plt.figure(num=title, figsize=(14, 5))

        # Plot test accuracy in the first subplot (position 2,3,1)
        plt.subplot(2, 3, 1)
        
        # Create a list of test round numbers (indices) for x-axis
        # This represents the round number for each test evaluation
        test_rounds = []
        # Iterate from 0 to the number of accuracy scores
        for i in range(len(accuracy)):
            # Append the round number (index) to the list
            test_rounds.append(i)
        
        # Set the x-axis to display only integer values (one tick per round)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot test accuracy scores with round numbers on x-axis
        plt.plot(test_rounds, accuracy, label='Test Accuracy', color='blue')
        # Set the x-axis label with the provided xlabel parameter
        plt.xlabel(xlabel)
        # Set the y-axis label to 'Accuracy'
        plt.ylabel('Accuracy')
        # Set the subplot title combining provided title with descriptive text
        plt.title('Test Accuracy ' + title)
        # Add a legend to identify the accuracy line
        plt.legend()

        # Plot test loss in the second subplot (position 2,3,2)
        plt.subplot(2, 3, 2)
        # Set the x-axis to display only integer values (one tick per round)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))   
        # Plot test loss scores with round numbers on x-axis
        plt.plot(test_rounds, loss, label='Test Loss', color='blue')
        
        # Set the x-axis label with the provided xlabel parameter
        plt.xlabel(xlabel)
        # Set the y-axis label to 'Loss'
        plt.ylabel('Loss')
        # Set the subplot title combining provided title with descriptive text
        plt.title('Test Loss ' + title)
        # Add a legend to identify the loss line
        plt.legend()

        # Adjust layout to prevent overlapping of subplots and labels
        plt.tight_layout()
        # Enable interactive mode: allows plots to be displayed without blocking execution
        # Multiple plots can be launched and viewed without blocking the program
        plt.ion()  

        # Display the complete figure with both subplots
        plt.show()
        
        
    def confusion_matrix(self, model, test_data, title : str):
        # Get predictions for all samples in the test data
        # model.predict returns logits or probabilities for each class
        y_pred = model.predict(test_data)
        # Convert predictions (probabilities) to class indices by taking the argmax (highest probability)
        # This gives us the predicted class for each sample
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Get the true class labels from the test data generator's classes attribute
        # test_data.classes contains the ground truth labels for evaluation
        true_classes = test_data.classes

        # Import the confusion_matrix function from sklearn.metrics
        # This will be used to compute the confusion matrix
        from sklearn.metrics import confusion_matrix
        # Compute the confusion matrix comparing true vs predicted labels
        # The confusion matrix shows correct and incorrect predictions per class
        conf_matrix = confusion_matrix(true_classes, y_pred_classes)

        # Create a new figure with custom title and size (14x14 inches) for the heatmap
        plt.figure(num=title, figsize=(14, 14))
        # Plot the confusion matrix in a specific subplot (position 2,3,3)
        plt.subplot(2, 3, 3)
        # Create a heatmap visualization of the confusion matrix
        # annot=True displays the actual count values in each cell
        # fmt='d' formats the values as integers
        # cmap='Blues' uses a blue color palette
        # square=True makes each cell square-shaped
        # xticklabels and yticklabels use the class names from the dataset
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True, 
                    xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
        # Set the x-axis label to describe predicted classes
        plt.xlabel('Predicted label')
        # Set the y-axis label to describe true classes
        plt.ylabel('Actual label')
        # Set the subplot title combining provided title with descriptive text
        plt.title('Confusion Matrix ' + title)
        # Enable interactive mode: allows plot to be displayed without blocking execution
        # Multiple plots can be launched and viewed without blocking the program
        plt.ion()  
        
        # Display the figure with the confusion matrix heatmap
        plt.show()