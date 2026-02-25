# Lab 3: Out-Of-Distribution Detection in Federated Learning
## Complete Implementation Guide

**Course:** Secure AI  
**Duration:** 1 Week (Groups of 2-4 students)  
**Objective:** Design and implement a local simulation framework for Federated Learning with an Out-of-Distribution (OOD) detection mechanism using Hyperdimensional Computing principles.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Architecture & Components](#architecture--components)
4. [Task 1: Setup & Run](#task-1-setup--run)
5. [Task 2: Implementing Phase 1 (Federated Learning)](#task-2-implementing-phase-1-federated-learning)
6. [Task 3: Implementing Phase 2 (OOD Detection)](#task-3-implementing-phase-2-ood-detection)
7. [Task 4: Experimentation](#task-4-experimentation)
8. [Submission Checklist](#submission-checklist)

---

## Project Overview

This lab focuses on implementing a **Federated Learning (FL)** framework that:
- Trains multiple local models on distributed datasets
- Aggregates models using a global model (central server)
- Detects Out-of-Distribution (OOD) attacks on local model updates
- Uses **Hyperdimensional Computing (HDC)** for feature projection and OOD detection

### Key Concepts

**Federated Learning**: A decentralized machine learning approach where:
- Multiple clients (local models) train on their local datasets
- The global server aggregates weights from all clients
- Data never leaves the client side

**Out-of-Distribution Detection**: Identifies compromised or adversarial local model updates before aggregation, protecting the global model from poisoning attacks.

**Hyperdimensional Computing**: Creates high-dimensional feature vectors for efficient OOD detection via:
- Projection matrices (random high-dimensional projections)
- Feature bundling (combining multiple features)
- Cosine similarity scoring

---

## Environment Setup

### Step 1: Install System Dependencies

**Linux/Ubuntu (required):**
```bash
sudo apt install python3.10
sudo apt install python3.10-venv
sudo apt install python3.10-dev
sudo apt-get install python3-tk
```

### Step 2: Navigate to Project Directory

```bash
cd /home/steven/Desktop/1-Lab3-AI/Lab/lab3/project
```

### Step 3: Install Python Dependencies

The project uses a Makefile for easy setup:

```bash
make setup
```

This command will:
- Create a Python 3.10 virtual environment in `./.env/.venv`
- Install all required packages from `python_requirements.txt`:
  - `tensorflow[and-cuda]` - Deep learning framework with GPU support
  - `opencv-python` - Image processing
  - `kagglehub` - Dataset downloading from Kaggle
  - `scikit-learn` - Machine learning utilities
  - `matplotlib` & `seaborn` - Visualization

**Note on CUDA:**
- If you don't have CUDA installed on your system, edit `python_requirements.txt` and change `tensorflow[and-cuda]` to just `tensorflow`
- This prevents compatibility issues with GPUs not present on your machine

### Step 4: Activate Virtual Environment (Optional)

```bash
source ./.env/.venv/bin/activate
```

---

## Architecture & Components

### Project Structure

```
lab3/project/
├── main.py                          # Entry point - defines two simulations
├── config.py                        # Configuration classes
├── Makefile                         # Build automation
├── python_requirements.txt          # Dependencies
│
├── model/
│   ├── model.py                     # CNN model definition & training
│   └── math/plot.py                 # Model plotting utilities
│
├── dataset/
│   ├── dataset.py                   # Dataset management & merging
│   ├── generator.py                 # Dataset generation from images
│   ├── download/                    # Dataset downloaders
│   │   ├── a_faces16000.py          # Face dataset (OOD)
│   │   ├── b_tumor3000.py           # Tumor dataset
│   │   ├── b_tumor4600.py           # Tumor dataset
│   │   ├── b_alzheimer5100.py       # Alzheimer's dataset (in-distribution)
│   │   ├── b_alzheimer5100_poisoned.py  # Poisoned Alzheimer's (OOD)
│   │   ├── l_pneumonia5200.py       # Pneumonia dataset (in-distribution)
│   │   └── utils/import_kaggle.py   # Kaggle API utilities
│   └── math/plot.py                 # Dataset plotting utilities
│
├── federated/
│   ├── federated.py                 # Federated Learning environment (Phase 1)
│   └── math/
│       ├── federated_math.py        # Aggregation & mathematical operations
│       └── plot.py                  # Federated plotting utilities
│
└── ood/
    ├── hdff.py                      # OOD detection using HDC (Phase 2)
    ├── VSA.py                       # Vector Symbolic Architectures
    └── math/score.py                # OOD scoring functions
```

### Key Configuration Classes

**ConfigModel**: Neural network training parameters
- `epochs`: Number of training iterations
- `activation`: Hidden layer activation function (e.g., 'relu')
- `activation_out`: Output layer activation ('softmax' for classification)
- `optimizer`: Training optimizer (e.g., 'adam')
- `loss`: Loss function ('categorical_crossentropy' for multi-class)

**ConfigDataset**: Image preprocessing & dataset splitting
- `batch_size`: Samples per training batch (default: 64)
- `image_size`: Target image dimensions (default: 256x256)
- `input_shape`: Model input shape (default: (256,256,1) for grayscale)
- `split`: Train/validation/test split ratio (default: 0.25)
- `number_of_classes`: Number of classification classes

**ConfigFederated**: Federated learning simulation parameters
- `rounds`: Number of federated training rounds
- `clients`: Total number of clients (global model + local models)
- `participants`: How many clients train each round
- `client_to_dataset`: Dataset allocation for each client
- `save`/`load`: Save/load model weights to/from disk

**ConfigOod**: Out-of-Distribution detection parameters
- `enabled`: Enable/disable OOD detection
- `hyper_size`: Hyperdimensional space size (e.g., 10,000)
- `id_client`: In-distribution client indices
- `ood_client`: Out-of-distribution dataset indices
- `ood_protection`: Enable filtering of OOD models
- `ood_protection_thres`: OOD detection threshold (0.0-1.0)

---

## Task 1: Setup & Run

### Objective
Get the project running with a single model (centralized) to verify environment setup.

### Steps

1. **Verify Setup**
   ```bash
   make run
   ```
   This runs the `ModelSimulation` class from `main.py` by default.

2. **What Happens**
   - The script downloads 4 medical image datasets from Kaggle (Tumor, Alzheimer's, Pneumonia)
   - Merges them into train/validation/test splits
   - Trains a simple CNN model for 5 epochs
   - Displays accuracy and loss metrics
   - Generates confusion matrix and training history plots
   - Press Enter when prompted to close plots and exit

3. **Expected Output**
   - Training progress with epoch-wise accuracy and loss
   - Test accuracy metrics
   - Plots saved to `./.env/.saved/`
   - Dataset cache location printed to terminal

### Troubleshooting

**Issue**: "Module not found" errors
- **Solution**: Ensure virtual environment is activated: `source ./.env/.venv/bin/activate`

**Issue**: CUDA/GPU errors
- **Solution**: Edit `python_requirements.txt`, change `tensorflow[and-cuda]` to `tensorflow`, then rerun `make setup`

**Issue**: Kaggle dataset download fails
- **Solution**: 
  - Install Kaggle API: `pip install kaggle`
  - Create account at kaggle.com, download API credentials (`kaggle.json`)
  - Place in `~/.kaggle/kaggle.json`
  - Run `chmod 600 ~/.kaggle/kaggle.json`

### Q&A
- **Q**: What's the difference between model training and federated learning?
  - **A**: Single model training trains on centralized data. Federated learning distributes data across clients, each training locally, then aggregating updates globally.

- **Q**: Why use grayscale (1 channel) vs RGB (3 channels)?
  - **A**: Medical images are often grayscale. For RGB, change `input_shape = (256,256,3)` in config.

---

## Task 2: Implementing Phase 1 (Federated Learning)

### Objective
Implement the federated learning pipeline: initialize clients, train locally, and aggregate globally.

### File to Modify
`lab3/project/federated/federated.py`

### Subtask 2.1: Initialize

**What to implement:**
Create local model instances for each client in the federation.

**Code Location**: `Federated.train_()` method

**Implementation Steps**:
1. For each round, retrieve the current global model weights
2. Initialize `self.federated_config.clients` number of local models
3. Distribute the global model weights to each client's local model
4. Assign datasets to clients based on `client_to_dataset` configuration

**Key Methods**:
- `self.init_model.model.get_weights()` - Get current global model weights
- `self.init_model.model.set_weights()` - Set weights for a local model
- `self.dataset.get(client_id)` - Get training data for a specific client

**Example Logic**:
```python
for client_id in range(self.federated_config.clients):
    # Create local model
    local_model = copy_model(self.init_model)
    
    # Set global weights to local model
    local_model.set_weights(self.init_model.get_weights())
    
    # Get client's dataset
    train_data, val_data, test_data = self.dataset.get(client_dataset_index)
```

### Subtask 2.2: Regression (Local Training Preparation)

**What to implement:**
Prepare local models before training by resetting to global weights if needed.

**Implementation Steps**:
1. For each participating client in the round, ensure the local model has the latest global weights
2. This is called "regression" - bringing local models back to global state before training

**Key Logic**:
```python
# Before training each local model, set it to current global weights
local_model.set_weights(global_model.get_weights())
```

### Subtask 2.3: Train (Local Training)

**What to implement:**
Train each local model on its own dataset for one epoch.

**Implementation Steps**:
1. Randomly select `self.federated_config.participants` clients from all clients
2. For each selected client:
   - Get its local model and assigned dataset
   - Train for one epoch (recommended: `epochs=1` in config)
   - Store the trained model weights

**Key Methods**:
- `local_model.train()` - Train on local data
- `random.sample(clients, participants)` - Select random participants

**Example Logic**:
```python
participants = random.sample(range(num_clients), self.federated_config.participants)
for client_id in participants:
    local_model.train(train_data, validation_data)
    trained_weights = local_model.get_weights()
```

### Subtask 2.4: Aggregation (Global Model Update)

**What to implement:**
Aggregate weights from all trained local models into the global model.

**Implementation Steps**:
1. Collect weights from all trained local models
2. Use weighted averaging or simple averaging to combine weights
3. Update the global model with aggregated weights
4. Store aggregated model for next round

**Key Methods**:
- `federated_math.aggregate_weights()` - Average multiple weight tensors
- `global_model.set_weights()` - Update global model

**Aggregation Formula**:
```
w_global = (w_client_1 + w_client_2 + ... + w_client_n) / n
```

**Example Logic**:
```python
collected_weights = [client1_weights, client2_weights, ...]
aggregated = federated_math.aggregate_weights(collected_weights)
global_model.set_weights(aggregated)
```

### Subtask 2.5: Result (Per-Round Metrics)

**What to implement:**
Evaluate and log metrics for each federated round.

**Implementation Steps**:
1. After aggregation each round, test the global model
2. Track accuracy, loss, and convergence metrics
3. Log results for plotting/analysis

**Key Methods**:
- `self.init_model.test()` - Evaluate on test set
- Store results for trend analysis

### Subtask 2.6: Saving & Loading Model

**What to implement:**
Save trained global model and load previously saved models.

**Implementation Steps**:
1. If `self.federated_config.save = True`, save global model weights after each round/simulation
2. If `self.federated_config.load = True`, load pre-trained global model from disk
3. Allow continuing simulation from a specific round

**Key Methods**:
- `model.save()` - Save model to disk
- `model.load()` - Load model from disk
- Save to path: `self.federated_config.path`

**File Format**: Save weights as NumPy arrays or TensorFlow SavedModel format

---

## Task 3: Implementing Phase 2 (OOD Detection)

### Objective
Implement Out-of-Distribution detection to identify and filter compromised/adversarial local model updates.

### File to Modify
`lab3/project/ood/hdff.py`

### Key Concepts

**Hyperdimensional Feature Framework (HDFF)**:
Uses high-dimensional random projections to detect anomalies in model weights.

**Workflow**:
1. **Projection**: Project model weights into high-dimensional space using random matrices
2. **Bundling**: Combine projected features using element-wise operations
3. **Scoring**: Compute cosine similarity between client updates and reference (in-distribution) model
4. **Filtering**: Exclude updates with scores below threshold

### Subtask 3.1: Creating Projection Matrices

**What to implement:**
Generate random high-dimensional projection matrices.

**Implementation Steps**:
1. Create `hyper_size` random projection matrices (one per model layer)
2. Each projection matrix dimensions: (original_layer_size, hyper_size)
3. Matrices should be normalized (e.g., using random normal distribution)

**Key Parameters**:
- `hyper_size`: Dimension of hyperdimensional space (default: 10,000)
- One matrix per trainable layer in the model

**Example Logic**:
```python
import numpy as np

projection_matrices = []
for layer in model.layers:
    if layer.weights:  # Skip layers without weights
        layer_shape = layer.weights[0].shape
        # Create random projection matrix
        proj_matrix = np.random.randn(layer_shape[0] * layer_shape[1], hyper_size)
        # Normalize
        proj_matrix = proj_matrix / np.linalg.norm(proj_matrix, axis=0)
        projection_matrices.append(proj_matrix)
```

### Subtask 3.2: Creating Feature Vectors

**What to implement:**
Convert model weights to high-dimensional feature vectors using projections.

**Implementation Steps**:
1. Flatten model weights from each layer
2. Project flattened weights using corresponding projection matrix
3. Normalize resulting feature vectors
4. Combine projections from all layers (bundling)

**Key Methods**:
- `np.dot(flattened_weights, projection_matrix)` - Project to high-dimensional space
- `np.linalg.norm()` - Normalize vectors

**Example Logic**:
```python
def create_feature_vector(model_weights, projection_matrices):
    features = np.zeros(projection_matrices[0].shape[1])  # hyper_size
    
    for i, layer_weights in enumerate(model_weights):
        # Flatten layer weights
        flattened = layer_weights.flatten()
        
        # Project to high-dim space
        projected = np.dot(flattened, projection_matrices[i])
        
        # Bundle (add to aggregate feature vector)
        features += projected
    
    # Normalize
    return features / np.linalg.norm(features)
```

### Subtask 3.3: Projection, Bundling & Cosine Similarity

**What to implement:**
Compute similarity scores between client updates and reference models.

**Implementation Steps**:
1. Create feature vectors for:
   - **In-distribution reference**: Average feature vector from known-good clients
   - **Test client**: Feature vector from client being evaluated
2. Compute cosine similarity between vectors
3. Score ranges from -1 to 1 (closer to 1 = more similar = in-distribution)

**Key Formula**:
```
cosine_similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```

**Example Logic**:
```python
from sklearn.metrics.pairwise import cosine_similarity

def score_client(client_features, reference_features):
    # Reshape for sklearn
    score = cosine_similarity(
        [client_features],
        [reference_features]
    )[0, 0]
    return score

# Use in OOD detection
for client_id in clients:
    client_vec = create_feature_vector(client_model.weights, projections)
    ref_vec = create_feature_vector(reference_model.weights, projections)
    score = score_client(client_vec, ref_vec)
    
    if score < ood_threshold:
        print(f"Client {client_id} is OOD!")
```

### Subtask 3.4: Integration into FL Environment

**What to implement:**
Integrate OOD detection into the federated learning pipeline.

**Implementation Steps**:
1. After each local training round, evaluate OOD score for each client
2. If OOD score < threshold AND `ood_protection = True`, exclude client from aggregation
3. Aggregate only weights from in-distribution clients
4. Log OOD detection results

**Integration Points** (in `federated.py`):
```python
# After local training, before aggregation
for client_id in trained_clients:
    if ood_config.enabled:
        score = hdff.score_client(client_weights, reference_weights)
        
        if score < ood_config.ood_protection_thres and ood_config.ood_protection:
            print(f"Excluding OOD client {client_id}")
            continue
    
    # Include client in aggregation
    weights_to_aggregate.append(client_weights)

# Aggregate only included weights
global_weights = aggregate(weights_to_aggregate)
```

**Key Objects**:
- `Hdff` class from `ood/hdff.py` - Main OOD detection module
- Methods: `score()`, `create_projections()`, `create_features()`

---

## Task 4: Experimentation

### Objective
Run 5-6 experiments demonstrating federated learning with OOD detection.

### Experiment Setup

Modify the `FederatedSimulation` class in `main.py` to run different scenarios.

### Base Configuration

**Standard Setup**:
- 5 clients total (1 global + 4 local)
- 4 in-distribution clients with medical images (Tumor, Alzheimer's, Pneumonia)
- 25 federated rounds
- OOD detection starts at round 26 (after pre-training)

### Experiment 1: OOD with Detection Disabled

**Configuration**:
```python
federated_config = ConfigFederated(
    rounds=25,
    clients=5,
    participants=4,
    ood_round=26,
    client_to_dataset=[[0,1,2,3],[0],[1],[2],[3], [4]]  # Client 5 = OOD data
)
ood_config = ConfigOod(
    enabled=False,  # OOD disabled
    ...
)
```

**Expected Result**:
- All clients aggregate, including OOD client
- Global model accuracy may degrade due to poisoned data
- Serves as baseline for comparison

### Experiment 2: Same OOD with Detection Enabled

**Configuration**:
```python
ood_config = ConfigOod(
    enabled=True,
    ood_protection=True,
    ood_protection_thres=0.7,
    id_client=[1,2,3,4],
    ood_client=[5]
)
```

**Expected Result**:
- OOD client detected and excluded from aggregation
- Global model maintains accuracy better than Experiment 1
- Shows effectiveness of OOD detection

### Experiment 3: New OOD Client with Detection

**Configuration**:
```python
client_to_dataset=[[0,1,2,3],[0],[1],[2],[3], [6]]  # Client 5 = different OOD
```

**Expected Result**:
- Different OOD dataset (e.g., Face images instead of poisoned medical)
- OOD detection still effective
- Demonstrates generalization

### Experiment 4: Mixed OOD Client

**Configuration**:
```python
# Half samples from poisoned Alzheimer's, half from ID Pneumonia
client_to_dataset=[[0,1,2,3],[0],[1],[2],[3], [4,3]]  # Mixed data
```

**Expected Result**:
- Harder to detect (partial OOD)
- May depend on threshold tuning
- Tests sensitivity of detection

### Experiment 5: Completely Different OOD Dataset

**Configuration**:
```python
client_to_dataset=[[0,1,2,3],[0],[1],[2],[3], [7]]  # Client 5 = Face dataset
```

**Expected Result**:
- Completely out-of-distribution (different image domain)
- Should be easily detected and excluded

### Experiment 6: Custom Scenario (Bonus)

Design your own experiment:
- Examples: Multiple OOD clients, varying threshold values, changing participation rates
- Document hypothesis, configuration, and results

### How to Run Experiments

1. **Uncomment FederatedSimulation** in `main.py`:
   ```python
   if __name__ == "__main__":
       # Comment out ModelSimulation
       # sim1 = ModelSimulation()
       # sim1.run()
       
       # Run Federated Simulation
       sim2 = FederatedSimulation()
       sim2.run()
   ```

2. **Modify configuration** for each experiment

3. **Run and collect results**:
   ```bash
   make run
   ```

4. **Record metrics**:
   - Global model accuracy per round
   - OOD detection rate (true positive rate)
   - Model accuracy with/without OOD protection

5. **Generate plots**:
   - Use `ModelPlot` and `FederatedPlot` classes to visualize trends
   - Compare accuracy curves across experiments

---

## Implementation Workflow Checklist

### Phase 1: Federated Learning (Task 2)

- [ ] **Initialize**: Create local models, distribute global weights, assign datasets
- [ ] **Regression**: Reset local models to global state each round
- [ ] **Train**: Locally train selected clients for one epoch
- [ ] **Aggregation**: Average weights from trained clients
- [ ] **Result**: Evaluate global model, log metrics
- [ ] **Save/Load**: Implement model persistence

### Phase 2: OOD Detection (Task 3)

- [ ] **Projections**: Create random high-dimensional matrices
- [ ] **Features**: Project model weights to feature vectors
- [ ] **Scoring**: Compute cosine similarity scores
- [ ] **Integration**: Filter OOD clients before aggregation

### Phase 3: Experiments (Task 4)

- [ ] **Experiment 1**: OOD disabled (baseline)
- [ ] **Experiment 2**: OOD enabled (same dataset)
- [ ] **Experiment 3**: OOD enabled (new OOD dataset)
- [ ] **Experiment 4**: Mixed OOD/ID data
- [ ] **Experiment 5**: Completely different OOD dataset
- [ ] **Experiment 6**: Custom scenario

---

## Submission Checklist

### Code Requirements

- [ ] **main.py**: Both `ModelSimulation` and `FederatedSimulation` runnable
- [ ] **config.py**: All configuration classes implemented
- [ ] **model/model.py**: CNN model definition and training/testing methods
- [ ] **dataset/dataset.py**: Dataset management, merging, client assignment
- [ ] **federated/federated.py**: Complete FL pipeline (initialize, train, aggregate)
- [ ] **federated/math/federated_math.py**: Weight aggregation functions
- [ ] **ood/hdff.py**: OOD detection using hyperdimensional computing
- [ ] **All plotting modules**: Visualization of results

### Documentation

- [ ] **README**: Setup and running instructions
- [ ] **Code comments**: Explain non-obvious logic
- [ ] **Experiment results**: Document all 5-6 experiments
- [ ] **Comparison table**: Results across experiments
- [ ] **Conclusions**: What was learned about OOD detection effectiveness

### Results & Analysis

- [ ] **Plots**: Accuracy curves, confusion matrices, OOD detection rates
- [ ] **Metrics comparison**: With/without OOD protection
- [ ] **Discussion**: How effective was OOD detection? Edge cases?
- [ ] **Future improvements**: How could this be enhanced?

---

## Useful Methods & Classes Reference

### Model Class (`model/model.py`)
```python
model = Model(model_config, dataset_config, plot_config)
model.train(train_data, validation_data)      # Train for N epochs
model.test(test_data)                         # Evaluate on test set
model.get_weights()                           # Get model weights
model.set_weights(weights)                    # Set model weights
model.plot_all(test_data, xlabel, title)      # Plot results
```

### Dataset Class (`dataset/dataset.py`)
```python
dataset = Dataset(datasets_list, dataset_config, plot_config)
train_data, val_data, test_data = dataset.mergeAll()  # Merge all
train_data, val_data, test_data = dataset.get(index)  # Get specific client data
```

### Federated Class (`federated/federated.py`)
```python
federated = Federated(dataset, model, federated_config, ood_config, dataset_config, plot_config)
federated.run()  # Run federated learning simulation
```

### Hdff Class (`ood/hdff.py`)
```python
hdff = Hdff(ood_config, model)
score = hdff.score(client_weights, reference_weights)  # Compute OOD score
```

---

## Helpful Tips

1. **Start with ModelSimulation**: Get the centralized training working first before federated learning
2. **Debug incrementally**: Test each federated component (init, train, aggregate) separately
3. **Use debug flags**: Set `debug=True` in configs to see detailed output
4. **Save intermediate results**: Use `save=True` to checkpoint models between runs
5. **Experiment with thresholds**: OOD detection effectiveness depends on `ood_protection_thres`
6. **Monitor memory**: Medical image datasets are large; monitor RAM usage
7. **Plot results**: Visual comparison is easier than reading numbers
8. **Document findings**: Keep notes on what works and what doesn't

---

## Troubleshooting Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory (OOM) | Reduce `batch_size` in config, or reduce number of images |
| Slow training | Reduce `image_size` from 256 to 128, or skip plotting |
| OOD never detected | Increase `ood_protection_thres` (closer to 1), verify projection size |
| Model not improving | Increase `epochs` for federated training, check data loading |
| Kaggle API fails | Ensure `kaggle.json` in `~/.kaggle/` with correct permissions |
| CUDA errors | Change `tensorflow[and-cuda]` to `tensorflow` in requirements |
| Virtual env issues | Delete `.env/` folder and rerun `make setup` |

---

## References & Further Reading

- Federated Learning: Communication-Efficient Learning of Deep Networks from Decentralized Data
- Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors
- Out-of-Distribution Detection: A Review and Practical Approaches

---

**Last Updated:** February 2026  
**Version:** 1.0
