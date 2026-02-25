# Import numpy for numerical array operations
import numpy as np
# Import TensorFlow for deep learning and tensor operations
import tensorflow as tf
# Import scikit-learn metrics for computing evaluation metrics (AUROC, AUPR)
from sklearn.metrics import roc_auc_score, average_precision_score

# Import OOD configuration class
from config import ConfigOod
# Import custom Dataset class for handling dataset information
from dataset.dataset import Dataset
# Import custom Model class for neural network models
from model.model import Model

class OodScore:
    # Docstring: Class for calculating AUROC (Area Under Receiver Operating Curve) and 
    # AUPR (Area Under Precision-Recall curve) scores based on model predictions and OOD/ID data
    """ Calculates AUROC and AUPR scores based on model, and ood / id data. """
    
    def __init__(self, ood_config: ConfigOod) -> None:
        # Store the OOD configuration object for later use
        self.ood_config = ood_config
        # Initialize empty list to store average AUROC scores for model predictions
        self.avg_auroc_model = []
        # Initialize empty list to store average AUPR scores for model predictions
        self.avg_aupr_model = []
        # Initialize empty list to store average AUROC scores for similarity metrics
        self.avg_auroc_similarity = []
        # Initialize empty list to store average AUPR scores for similarity metrics
        self.avg_aupr_similarity = []
    
    def get_softmax_scores(self, model, generator):
        """Computes max softmax confidence scores for a dataset."""
        # Initialize empty list to store softmax confidence scores
        scores = []
        # Initialize empty list to store ground truth labels
        ground_truth = []
        # Iterate through batches of images and labels from the data generator
        for images, labels in generator:  
            # Get model predictions (logits) for the current batch of images
            logits = model.predict(images, verbose=0)
            # Convert logits to softmax probabilities to get confidence scores
            softmax_scores = tf.nn.softmax(logits, axis=1)  
            # Get the index of the maximum probability for each sample (predicted class)
            max_scores = tf.argmax(softmax_scores, axis=1).numpy()  
            # Extend the scores list with the predicted class indices for this batch
            scores.extend(max_scores)
            
            # Convert one-hot encoded labels to class indices by taking argmax
            numeric_labels = tf.argmax(labels, axis=1).numpy()
            # Extend the ground_truth list with the true class indices for this batch
            ground_truth.extend(numeric_labels)
            
            # Stop when all images are processed (check if we've reached total number of samples)
            if len(scores) >= generator.n:
                # Break the loop once all data has been processed
                break
        
        # Return the list of predicted scores and ground truth labels
        return scores, ground_truth
    
    def compute_auroc_model(self, model, generator, ood_config: ConfigOod):
        # Get softmax scores and ground truth labels from the model and data generator
        model_scores, ground_truth = self.get_softmax_scores(model, generator)
        
        # Debug prints (commented out but available for troubleshooting)
        #print(f"Model Scores: {model_scores}")
        #print(f"Ground Truth: {ground_truth}")
        
        # Compute AUROC score using scikit-learn's roc_auc_score function
        # AUROC measures the model's ability to distinguish between positive and negative classes
        auroc = roc_auc_score(ground_truth, model_scores)
        # Append the current AUROC score to the list for averaging later
        self.avg_auroc_model.append(auroc)

        # Print a blank line for formatting
        print()
        # Print header label for model scores AUROC results
        print("Model scores AUROC")
        # Print the AUROC score for the current evaluation
        print(f"\nAUROC SCORE = {auroc:.4f}")
        
        # Compute the average of all AUROC scores collected so far
        avg_auroc = np.mean(self.avg_auroc_model)
        # Print the average AUROC score across all evaluations
        print(f"AUROC SCORE (AVG) = {avg_auroc:.4f}\n")
        
        # Return the average AUROC score
        return avg_auroc
    
    def compute_aupr_model(self, model, generator, ood_config: ConfigOod):
        # Get softmax scores and ground truth labels from the model and data generator
        model_scores, ground_truth = self.get_softmax_scores(model, generator)
        
        # Debug prints (commented out but available for troubleshooting)
        #print(f"Model Scores: {model_scores}")
        #print(f"Ground Truth: {ground_truth}")

        # Compute AUPR score using scikit-learn's average_precision_score function
        # AUPR measures the area under the precision-recall curve for model predictions
        aupr = average_precision_score(ground_truth, model_scores)
        # Append the current AUPR score to the list for averaging later
        self.avg_aupr_model.append(aupr)
        
        # Print header label for model scores AUPR results
        print("Model scores AUPR")
        # Print the AUPR score for the current evaluation
        print(f"\nAUPR SCORE = {aupr:.4f}")
        
        # Compute the average of all AUPR scores collected so far
        avg_aupr = np.mean(self.avg_aupr_model)
        # Print the average AUPR score across all evaluations
        print(f"AUPR SCORE (AVG) = {avg_aupr:.4f}\n")
        
        # Return the average AUPR score
        return avg_aupr
    
    def compute_auroc_similarity(self, similarity, ood_config: ConfigOod):
        """_summary_

        Args:
            similarity (list): list with similairty scores of model, from HDFF.
            ood_config (ConfigOod): _description_

        Returns:
            list: AUROC score, averaging siimilarity scores. 
        """
        # Convert similarity list to numpy array for easier manipulation
        similarity_scores = np.array(similarity)
        
        # Create ground truth labels: 1 for ID (in-distribution) clients, 0 for OOD clients
        # ID clients are marked as 1 (positive class), OOD clients are marked as 0 (negative class)
        ground_truth = [1] * len(ood_config.id_client) + [0] * len(ood_config.ood_client)
        # Convert ground_truth list to numpy array
        ground_truth = np.array(ground_truth)
        
        # Debug prints (commented out but available for troubleshooting)
        # print(f"Similarity Scores: {similarity_scores}")
        # print(f"Ground Truth: {ground_truth}")
        
        # Compute AUROC score using similarity scores and ground truth labels
        # This measures how well the similarity metric distinguishes ID from OOD models
        auroc = roc_auc_score(ground_truth, similarity_scores)
        # Append the current AUROC score to the list for averaging later
        self.avg_auroc_similarity.append(auroc)
        
        # Print header label for similarity scores AUROC results
        print("Similarity scores AUROC")
        # Print the AUROC score for the current evaluation
        print(f"\nAUROC SCORE = {auroc:.4f}")
        
        # Compute the average of all AUROC scores collected so far
        avg_auroc = np.mean(self.avg_auroc_similarity)
        # Print the average AUROC score across all evaluations
        print(f"AUROC SCORE (AVG) = {avg_auroc:.4f}\n")
        
        # Return the average AUROC score
        return avg_auroc
        
    def compute_aupr_similarity(self, similarity, ood_config: ConfigOod):
        # Convert similarity list to numpy array for easier manipulation
        similarity_scores = np.array(similarity)
        # Create ground truth labels: 1 for ID (in-distribution) clients, 0 for OOD clients
        # ID clients are marked as 1 (positive class), OOD clients are marked as 0 (negative class)
        ground_truth = [1] * len(ood_config.id_client) + [0] * len(ood_config.ood_client)
        # Convert ground_truth list to numpy array
        ground_truth = np.array(ground_truth)

        # Debug prints (commented out but available for troubleshooting)
        # print(f"Similarity Scores: {similarity_scores}")
        # print(f"Ground Truth: {ground_truth}")

        # Compute AUPR score using similarity scores and ground truth labels
        # This measures the area under the precision-recall curve for OOD detection
        aupr = average_precision_score(ground_truth, similarity_scores)
        # Append the current AUPR score to the list for averaging later
        self.avg_aupr_similarity.append(aupr)

        # Print header label for similarity scores AUPR results
        print("Similarity scores AUPR")
        # Print the AUPR score for the current evaluation
        print(f"\nAUPR SCORE = {aupr:.4f}")
        
        # Compute the average of all AUPR scores collected so far
        avg_aupr = np.mean(self.avg_aupr_similarity)
        # Print the average AUPR score across all evaluations
        print(f"AUPR SCORE (AVG) = {avg_aupr:.4f}\n")
        
        # Return the average AUPR score
        return avg_aupr