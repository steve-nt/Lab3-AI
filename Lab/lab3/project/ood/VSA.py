# Import TensorFlow library for tensor operations and neural network operations
import tensorflow as tf

# Define the Vsa class which implements Vector Symbolic Architectures (VSA) operations
class Vsa:
    """ VSA/HDC mathematics form the backbone of the ood detection approach.
    References:
        - https://arxiv.org/abs/2112.05341
        - https://github.com/SamWilso/HDFF_Official
        
        However code is re-written to fit tensorflow and this project structure.
    """
    # Class-level debug flag initialized to False
    debug = False
    
    def __init__(self, debug) -> None:
        # Initialize the class with a debug flag to enable/disable debug output
        self.debug = debug
    
    def _dim_check(self, x, y):
        # Private helper method to ensure both tensors have at least 2 dimensions (batch dimension)
        # Check if x has less than 2 dimensions (e.g., is a 1D vector)
        if len(tf.shape(x)) < 2:
            # Add a batch dimension to x by expanding dims at axis 0 (prepending batch dimension)
            x = tf.expand_dims(x, axis=0)
        # Check if y has less than 2 dimensions (e.g., is a 1D vector)
        if len(tf.shape(y)) < 2:
            # Add a batch dimension to y by expanding dims at axis 0 (prepending batch dimension)
            y = tf.expand_dims(y, axis=0)
        # Return both tensors with guaranteed batch dimensions
        return x, y

    def bundle(self, x, y) -> tf.Tensor:
        # Bundle operation: combines two vectors using element-wise addition (superposition in VSA terms)
        # Ensure both vectors have the required dimensionality (batch_size, dimensions)
        x, y = self._dim_check(x, y)
        # Return the element-wise sum of the two tensors (bundling/superposition operation)
        return x + y
	
    def bulk_bundle(self, x) -> tf.Tensor:
        # Bulk bundle operation: sums all vectors along the batch axis (dimension 0)
        # This performs bundling (superposition) across all samples in a batch
        return tf.reduce_sum(x, axis=0)
    
    def bind(self, x, y) -> tf.Tensor:
        # Bind operation: combines two vectors using element-wise multiplication (binding in VSA terms)
        # Ensure both vectors have the required dimensionality (batch_size, dimensions)
        x, y = self._dim_check(x, y)
        # Return the element-wise product of the two tensors (binding operation)
        return x * y

    def norm(self, tensor):
        # Normalize the tensor using L2 normalization (Euclidean norm)
        # Normalize along axis 1 (feature/dimension axis), preserving batch dimension
        return tf.nn.l2_normalize(tensor, axis=1)

    def similarity(self, x, y):
        # Compute cosine similarity between two sets of vectors
        # Both should have shape (n_samples, hyper_dim) - batch dimension plus feature dimensions
        # Ensure both tensors have the required dimensionality
        x, y = self._dim_check(x, y)
        # L2 normalize both tensors along the feature axis to unit vectors
        x, y = self.norm(x), self.norm(y)
        # Compute similarity as the dot product between x and y^T (matrix multiplication with transposed y)
        # Result shape: (n_samples_x, n_samples_y) representing pairwise cosine similarity
        return tf.linalg.matmul(x, y, transpose_b=True)
    
    def euclidean_distance(self, x, y):
        """
        Compute the pairwise Euclidean distance between two tensors.

        Args:
            x (tf.Tensor): Tensor of shape (n_samples, n_features).
            y (tf.Tensor): Tensor of shape (m_samples, n_features).

        Returns:
            tf.Tensor: Pairwise Euclidean distances of shape (n_samples, m_samples).
        """
        # Ensure both tensors have the required dimensionality with batch dimensions
        x, y = self._dim_check(x, y)
        # Compute squared differences by expanding dimensions for broadcasting:
        # x: (n_samples, 1, n_features), y: (1, m_samples, n_features)
        # Result: (n_samples, m_samples, n_features) containing all pairwise differences
        squared_diff = tf.expand_dims(x, axis=1) - tf.expand_dims(y, axis=0)
        # Sum the squared differences along the last axis (feature axis) to get squared distances
        # Result shape: (n_samples, m_samples)
        squared_distances = tf.reduce_sum(tf.square(squared_diff), axis=-1)
        # Take the square root of squared distances to get actual Euclidean distances
        distances = tf.sqrt(squared_distances)

        # Return the pairwise Euclidean distance matrix of shape (n_samples, m_samples)
        return distances

# Main execution block: runs only when this file is executed directly (not when imported)
if __name__ == "__main__": 
    # Create an instance of the Vsa class with debug mode enabled
    hdff = Vsa(debug=True)
