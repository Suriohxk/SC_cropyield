# 📚 Complete Documentation

## Table of Contents
1. [Neural Network Implementations](#neural-network-implementations)
2. [Data Generation](#data-generation)
3. [Application Architecture](#application-architecture)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Code Examples](#code-examples)

---

## Neural Network Implementations

### 1. McCulloch-Pitts Neuron

**Purpose**: Demonstrates the foundational model of artificial neurons (1943)

**Key Characteristics:**
- Binary threshold activation
- Fixed weights (no learning)
- Boolean logic implementation

**Implementation Details:**
```python
class McCullochPittsNeuron:
    def __init__(self, n_inputs: int, threshold: float = 0.0):
        self.weights = np.ones(n_inputs)
        self.threshold = threshold

    def activate(self, inputs: np.ndarray) -> int:
        weighted_sum = np.dot(inputs, self.weights)
        return 1 if weighted_sum >= self.threshold else 0
```

**Activation Function:**
```
output = 1 if Σ(wi × xi) ≥ θ else 0
```
where:
- wi = weight for input i
- xi = input value i
- θ = threshold

**Applications in App:**
- AND gate: weights=[1,1], threshold=2
- OR gate: weights=[1,1], threshold=1
- NAND gate: weights=[-1,-1], threshold=-1
- NOR gate: weights=[-1,-1], threshold=-2

---

### 2. Single-Layer Perceptron

**Purpose**: Introduces learning capability via Delta rule

**Key Characteristics:**
- Adaptive weights
- Binary classification
- Linear decision boundary
- Cannot solve XOR (non-linearly separable)

**Implementation Details:**
```python
class SingleLayerPerceptron:
    def __init__(self, n_inputs: int, learning_rate: float = 0.01):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
```

**Delta Learning Rule:**
```
Δw = η × (target - output) × input
Δb = η × (target - output)
```
where:
- η = learning rate
- target = desired output
- output = actual prediction

**Training Process:**
1. Initialize random weights
2. For each training example:
   - Compute prediction: ŷ = activate(Σ(wi × xi) + b)
   - Calculate error: e = y - ŷ
   - Update weights: wi ← wi + η × e × xi
   - Update bias: b ← b + η × e
3. Repeat for multiple epochs

**Linear Separability:**
- Can solve: AND, OR, NAND, NOR
- Cannot solve: XOR, XNOR
- Decision boundary is a hyperplane

---

### 3. Self-Organizing Map (SOM)

**Purpose**: Unsupervised clustering with topology preservation

**Key Characteristics:**
- Winner-Take-All competitive learning
- Neighborhood function
- Topology-preserving dimensionality reduction
- No labeled data required

**Implementation Details:**
```python
class SelfOrganizingMap:
    def __init__(self, grid_size: Tuple[int, int], input_dim: int,
                 learning_rate: float = 0.5):
        self.grid_size = grid_size
        self.weights = np.random.randn(grid_size[0], grid_size[1],
                                       input_dim) * 0.1
```

**Training Algorithm:**

1. **Initialization**: Random weight vectors for each neuron

2. **Competition** (Winner-Take-All):
   - For each input vector x:
   - Find Best Matching Unit (BMU): neuron with closest weights
   ```
   BMU = argmin ||x - wi||²
   ```

3. **Cooperation** (Neighborhood Function):
   - Calculate influence based on distance from BMU
   ```
   h(d) = exp(-d² / (2σ²))
   ```
   where:
   - d = distance from BMU
   - σ = neighborhood radius (decreases over time)

4. **Adaptation** (Weight Update):
   ```
   wi(t+1) = wi(t) + η(t) × h(d) × (x - wi(t))
   ```
   where:
   - η(t) = learning rate at time t (decreases)
   - h(d) = neighborhood influence

**U-Matrix Visualization:**
- Shows average distance between neighboring neurons
- Dark regions = cluster centers
- Bright regions = cluster boundaries

**Training Schedule:**
```python
radius(t) = initial_radius × exp(-t / total_epochs)
learning_rate(t) = initial_lr × exp(-t / total_epochs)
```

---

### 4. Multi-Layer Perceptron (MLP)

**Purpose**: Non-linear regression with backpropagation

**Architecture:**
```
Input Layer (6 neurons)
    ↓
Hidden Layer (configurable, sigmoid activation)
    ↓
Output Layer (1 neuron, linear activation)
```

**Implementation Details:**
```python
class MLPRegressor:
    def __init__(self, input_size: int, hidden_size: int,
                 learning_rate: float = 0.01):
        # He initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
                                   * np.sqrt(2.0 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, 1)
                                    * np.sqrt(2.0 / hidden_size)
```

**Forward Propagation:**
```
1. Hidden Layer:
   h_input = X · W1 + b1
   h_output = σ(h_input)

2. Output Layer:
   y_pred = h_output · W2 + b2

where σ(x) = 1 / (1 + e^(-x))  [sigmoid]
```

**Backpropagation Algorithm:**

1. **Forward Pass**: Calculate predictions
2. **Compute Output Error**:
   ```
   δ_output = y_pred - y_true
   ```

3. **Compute Hidden Error**:
   ```
   δ_hidden = (δ_output · W2^T) ⊙ σ'(h_output)
   ```
   where σ'(x) = σ(x) × (1 - σ(x))

4. **Update Weights** (Gradient Descent):
   ```
   W2 ← W2 - η × (h_output^T · δ_output) / m
   b2 ← b2 - η × Σ(δ_output) / m

   W1 ← W1 - η × (X^T · δ_hidden) / m
   b1 ← b1 - η × Σ(δ_hidden) / m
   ```
   where:
   - η = learning rate
   - m = batch size

**Loss Function** (Mean Squared Error):
```
MSE = (1/m) × Σ(y_pred - y_true)²
```

**Activation Functions:**
- Hidden layer: Sigmoid (smooth, differentiable)
- Output layer: Linear (for regression)

---

### 5. Hebbian Learning (Optional)

**Principle**: "Neurons that fire together, wire together"

**Weight Update Rule:**
```
Δwij = η × xi × yj
```

**Implementation**:
- Used for weight pre-initialization
- Alternative to random initialization
- Can speed up convergence

---

## Data Generation

### Synthetic Crop Dataset

**Features (6 inputs):**
1. Rainfall (mm): 400-1200
2. Temperature (°C): 15-35
3. Soil pH: 5.5-8.0
4. Nitrogen (kg/ha): 20-120
5. Phosphorus (kg/ha): 10-80
6. Potassium (kg/ha): 10-80

**Target:**
- Yield (tons/ha): 1.5-9.0

**Generation Formula:**
```python
yield = base_yield
        - 0.003 × (rainfall - 800)² / 100
        - 0.01 × (temperature - 25)² / 10
        - 0.5 × (pH - 6.5)²
        + 0.02 × nitrogen
        + 0.015 × phosphorus
        + 0.01 × potassium
        + 0.0001 × nitrogen × phosphorus
        + noise
```

**Characteristics:**
- Non-linear relationships (quadratic penalties)
- Optimal values: rainfall=800, temp=25, pH=6.5
- Fertilizer interactions (N-P-K synergy)
- Gaussian noise: N(0, 0.3)

**Data Normalization:**
```python
X_normalized = (X - mean) / std
```

This ensures:
- Mean = 0, Standard Deviation = 1
- Stable gradient descent
- Faster convergence

---

## Application Architecture

### Multi-Page Structure

```
app.py
├── Session State Management
│   ├── data: DataFrame (synthetic crop data)
│   ├── models_trained: Boolean
│   ├── mlp: Trained MLP model
│   ├── som: Trained SOM model
│   └── normalization_params: mean, std
│
├── Page 1: Home (🏠)
│   └── Project overview & concepts
│
├── Page 2: Data Exploration (📊)
│   ├── Statistics
│   ├── Distributions
│   └── Correlations
│
├── Page 3: McCulloch-Pitts & Perceptron (🧠)
│   ├── Logic gate demos
│   ├── Perceptron training
│   └── Linear separability visualization
│
├── Page 4: SOM Clustering (🗺️)
│   ├── SOM training
│   ├── U-Matrix
│   └── Cluster analysis
│
├── Page 5: MLP Training (🤖)
│   ├── MLP training with backpropagation
│   ├── Loss curves
│   └── Performance metrics
│
└── Page 6: Crop Yield Predictor (🌾)
    ├── Interactive prediction
    ├── Recommendations
    └── Sample scenarios
```

### Data Flow

```
User Input (sliders)
    ↓
Normalization
    ↓
MLP Forward Pass
    ↓
Denormalization
    ↓
Prediction + Interpretation
```

---

## Mathematical Foundations

### Gradient Descent

**Objective**: Minimize loss function J(θ)

**Update Rule:**
```
θ ← θ - η × ∇J(θ)
```

**Chain Rule** (for backpropagation):
```
∂J/∂w1 = (∂J/∂y) × (∂y/∂h) × (∂h/∂w1)
```

### Activation Functions

**Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))

Derivative: σ'(x) = σ(x) × (1 - σ(x))
```

**Properties:**
- Range: (0, 1)
- Smooth and differentiable
- Can cause vanishing gradients

**Step Function** (Perceptron):
```
f(x) = 1 if x ≥ 0 else 0
```

**Properties:**
- Binary output
- Not differentiable
- Simple threshold

### Distance Metrics

**Euclidean Distance** (SOM):
```
d(x, y) = √(Σ(xi - yi)²)
```

**Mean Squared Error** (MLP):
```
MSE = (1/n) × Σ(yi_pred - yi_true)²
```

---

## Code Examples

### Training the MLP

```python
# Prepare data
X = df.drop('Yield (tons/ha)', axis=1).values
y = df['Yield (tons/ha)'].values

# Normalize
X_norm, X_mean, X_std = normalize_data(X)
y_norm, y_mean, y_std = normalize_data(y.reshape(-1, 1))

# Create and train MLP
mlp = MLPRegressor(input_size=6, hidden_size=12, learning_rate=0.01)
mlp.train(X_norm, y_norm.flatten(), epochs=1500)

# Make predictions
y_pred_norm = mlp.predict(X_norm)
y_pred = denormalize_data(y_pred_norm.reshape(-1, 1), y_mean, y_std)
```

### Training the SOM

```python
# Create SOM
som = SelfOrganizingMap(grid_size=(5, 5), input_dim=6, learning_rate=0.5)

# Train
som.train(X_normalized, epochs=200)

# Get clusters
cluster_map = som.get_cluster_map(X_normalized)

# Visualize U-Matrix
umatrix = som.get_umatrix()
```

### Making Predictions

```python
# New input
input_features = np.array([[800, 25, 6.5, 80, 50, 50]])

# Normalize
input_norm = (input_features - X_mean) / X_std

# Predict
prediction_norm = mlp.predict(input_norm)

# Denormalize
prediction = denormalize_data(prediction_norm.reshape(-1, 1),
                             y_mean, y_std)[0, 0]
```

---

## Performance Optimization

### Tips for Better Training

1. **Learning Rate Selection:**
   - Too high: Unstable, oscillation
   - Too low: Slow convergence
   - Recommended: 0.01 for MLP, 0.5 for SOM

2. **Hidden Layer Size:**
   - Too small: Underfitting
   - Too large: Overfitting
   - Rule of thumb: (input_size + output_size) / 2

3. **Training Epochs:**
   - Monitor loss curve
   - Stop when plateauing
   - Use early stopping if overfitting

4. **Data Normalization:**
   - Always normalize inputs
   - Prevents gradient issues
   - Speeds up convergence

### Evaluation Metrics

**Mean Squared Error (MSE):**
- Penalizes large errors heavily
- Units: (tons/ha)²

**Root Mean Squared Error (RMSE):**
- Same units as target
- Interpretable
- RMSE = √MSE

**Mean Absolute Error (MAE):**
- Robust to outliers
- Average absolute difference
- MAE = (1/n) × Σ|yi_pred - yi_true|

---

## Advanced Topics

### Hybrid SOM + MLP Approach

**Concept**: Use SOM clusters as additional features for MLP

**Process:**
1. Train SOM on input features
2. Get cluster assignments for each sample
3. One-hot encode cluster IDs
4. Concatenate with original features
5. Train MLP on augmented features

**Benefits:**
- Captures spatial relationships
- Adds non-linear feature engineering
- Can improve prediction accuracy

### Avoiding Overfitting

**Techniques:**
1. Train/test split (80/20)
2. Monitor both training and validation loss
3. Early stopping
4. Regularization (L1/L2) - can be added
5. Dropout (can be implemented)

### Hyperparameter Tuning

**Grid Search Example:**
```python
best_rmse = float('inf')
best_params = {}

for hidden_size in [8, 12, 16, 20]:
    for lr in [0.001, 0.01, 0.1]:
        mlp = MLPRegressor(input_size=6, hidden_size=hidden_size,
                          learning_rate=lr)
        mlp.train(X_train, y_train, epochs=1000)
        predictions = mlp.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {'hidden_size': hidden_size, 'lr': lr}
```

---

## Troubleshooting Common Issues

### Issue: Perceptron not converging on XOR
**Reason**: XOR is not linearly separable
**Solution**: This is expected! Demonstrates limitation of single-layer networks

### Issue: MLP loss not decreasing
**Possible causes:**
1. Learning rate too high → Reduce to 0.001
2. Bad initialization → Retrain with different seed
3. Data not normalized → Check normalization

### Issue: SOM clusters too spread out
**Solution**:
- Increase grid size
- Train for more epochs
- Increase learning rate

### Issue: Predictions all similar
**Possible causes:**
1. Model underfitting → Add hidden neurons
2. Stuck in local minimum → Increase learning rate
3. Data not diverse → Check data generation

---

## References & Further Reading

### Foundational Papers

1. McCulloch & Pitts (1943): "A logical calculus of the ideas immanent in nervous activity"
2. Rosenblatt (1958): "The perceptron: A probabilistic model"
3. Kohonen (1982): "Self-organized formation of topologically correct feature maps"
4. Rumelhart et al. (1986): "Learning representations by back-propagating errors"

### Books

1. "Neural Networks and Learning Machines" - Simon Haykin
2. "Pattern Recognition and Machine Learning" - Christopher Bishop
3. "Deep Learning" - Goodfellow, Bengio, Courville

### Online Resources

1. Stanford CS229: Machine Learning
2. MIT 6.034: Artificial Intelligence
3. 3Blue1Brown: Neural Networks series

---

**Built as Soft Computing Course Project**
All implementations from scratch using NumPy
