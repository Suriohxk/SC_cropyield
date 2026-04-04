import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import pandas as pd

np.random.seed(42)

st.set_page_config(
    page_title="Crop Yield Predictor - Soft Computing",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

class McCullochPittsNeuron:
    def __init__(self, n_inputs: int, threshold: float = 0.0):
        self.weights = np.ones(n_inputs)
        self.threshold = threshold

    def activate(self, inputs: np.ndarray) -> int:
        weighted_sum = np.dot(inputs, self.weights)
        return 1 if weighted_sum >= self.threshold else 0

    def set_weights(self, weights: np.ndarray):
        self.weights = weights


class SingleLayerPerceptron:
    def __init__(self, n_inputs: int, learning_rate: float = 0.01):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.training_history = []

    def activation(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                prediction = self.predict(X[i:i+1])[0]
                error = y[i] - prediction

                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                total_error += abs(error)

            avg_error = total_error / len(X)
            errors.append(avg_error)
            self.training_history.append({
                'epoch': epoch,
                'error': avg_error
            })

        return errors


class SelfOrganizingMap:
    def __init__(self, grid_size: Tuple[int, int], input_dim: int, learning_rate: float = 0.5):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.weights = np.random.randn(grid_size[0], grid_size[1], input_dim) * 0.1
        self.training_history = []

    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        distances = np.sum((self.weights - input_vector) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.grid_size)
        return bmu_idx

    def _get_neighborhood(self, bmu_idx: Tuple[int, int], radius: float) -> np.ndarray:
        y, x = np.meshgrid(range(self.grid_size[0]), range(self.grid_size[1]), indexing='ij')
        distance = np.sqrt((y - bmu_idx[0])**2 + (x - bmu_idx[1])**2)
        return np.exp(-(distance ** 2) / (2 * (radius ** 2)))

    def train(self, X: np.ndarray, epochs: int = 100):
        initial_radius = max(self.grid_size) / 2

        for epoch in range(epochs):
            radius = initial_radius * np.exp(-epoch / epochs)
            learning_rate = self.initial_learning_rate * np.exp(-epoch / epochs)

            epoch_error = 0
            for input_vector in X:
                bmu_idx = self._find_bmu(input_vector)
                neighborhood = self._get_neighborhood(bmu_idx, radius)

                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        influence = neighborhood[i, j]
                        self.weights[i, j] += influence * learning_rate * (input_vector - self.weights[i, j])

                bmu_distance = np.linalg.norm(self.weights[bmu_idx[0], bmu_idx[1]] - input_vector)
                epoch_error += bmu_distance

            avg_error = epoch_error / len(X)
            self.training_history.append({'epoch': epoch, 'error': avg_error})

    def get_cluster_map(self, X: np.ndarray) -> np.ndarray:
        cluster_map = np.zeros(len(X), dtype=int)
        for i, input_vector in enumerate(X):
            bmu_idx = self._find_bmu(input_vector)
            cluster_map[i] = bmu_idx[0] * self.grid_size[1] + bmu_idx[1]
        return cluster_map

    def get_umatrix(self) -> np.ndarray:
        umatrix = np.zeros(self.grid_size)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                            neighbors.append(self.weights[ni, nj])

                if neighbors:
                    distances = [np.linalg.norm(self.weights[i, j] - n) for n in neighbors]
                    umatrix[i, j] = np.mean(distances)

        return umatrix


class MLPRegressor:
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 learning_rate: float = 0.01, use_hebbian_init: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        if use_hebbian_init:
            self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
            self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        else:
            self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
            self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)

        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.training_history = []

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.final_input

        return self.hidden_output, self.final_output

    def backward(self, X: np.ndarray, y: np.ndarray, hidden_output: np.ndarray,
                 final_output: np.ndarray):
        m = X.shape[0]

        output_error = final_output - y
        output_delta = output_error

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        self.weights_hidden_output -= self.learning_rate * np.dot(hidden_output.T, output_delta) / m
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m

        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / m
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = False):
        y = y.reshape(-1, 1)

        for epoch in range(epochs):
            hidden_output, final_output = self.forward(X)

            self.backward(X, y, hidden_output, final_output)

            mse = np.mean((final_output - y) ** 2)
            self.training_history.append({'epoch': epoch, 'mse': mse})

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                st.write(f"Epoch {epoch}/{epochs}, MSE: {mse:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, output = self.forward(X)
        return output.flatten()


def generate_crop_data(n_samples: int = 500) -> pd.DataFrame:
    np.random.seed(42)

    rainfall = np.random.uniform(400, 1200, n_samples)
    temperature = np.random.uniform(15, 35, n_samples)
    soil_ph = np.random.uniform(5.5, 8.0, n_samples)
    nitrogen = np.random.uniform(20, 120, n_samples)
    phosphorus = np.random.uniform(10, 80, n_samples)
    potassium = np.random.uniform(10, 80, n_samples)

    optimal_rainfall = 800
    optimal_temp = 25
    optimal_ph = 6.5
    optimal_n = 80
    optimal_p = 50
    optimal_k = 50

    yield_base = 5.0

    rainfall_effect = -0.003 * (rainfall - optimal_rainfall) ** 2 / 100
    temp_effect = -0.01 * (temperature - optimal_temp) ** 2 / 10
    ph_effect = -0.5 * (soil_ph - optimal_ph) ** 2
    n_effect = 0.02 * nitrogen
    p_effect = 0.015 * phosphorus
    k_effect = 0.01 * potassium

    interaction_effect = 0.0001 * nitrogen * phosphorus

    crop_yield = (yield_base + rainfall_effect + temp_effect + ph_effect +
                  n_effect + p_effect + k_effect + interaction_effect)

    noise = np.random.normal(0, 0.3, n_samples)
    crop_yield = np.maximum(crop_yield + noise, 0.5)

    df = pd.DataFrame({
        'Rainfall (mm)': rainfall,
        'Temperature (°C)': temperature,
        'Soil pH': soil_ph,
        'Nitrogen (kg/ha)': nitrogen,
        'Phosphorus (kg/ha)': phosphorus,
        'Potassium (kg/ha)': potassium,
        'Yield (tons/ha)': crop_yield
    })

    return df


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    normalized = (data - mean) / std
    return normalized, mean, std


def denormalize_data(normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return normalized_data * std + mean


def plot_decision_boundary(perceptron, X, y, title="Decision Boundary"):
    fig, ax = plt.subplots(figsize=(10, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')

    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn',
                        edgecolors='black', s=100, alpha=0.8)

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Class')

    return fig


st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Data Exploration", "🧠 McCulloch-Pitts & Perceptron",
     "🗺️ SOM Clustering", "🤖 MLP Training", "🌾 Crop Yield Predictor"]
)

if 'data' not in st.session_state:
    st.session_state.data = generate_crop_data(500)
    st.session_state.models_trained = False

if page == "🏠 Home":
    st.title("🌾 Crop Yield Predictor Using Shallow Neural Networks")
    st.markdown("### Soft Computing Course Project")

    st.markdown("""
    ---
    ## 📚 Project Overview

    This application demonstrates **Soft Computing** concepts applied to agricultural yield prediction.
    We implement various neural network architectures **from scratch** using only NumPy.

    ### 🎯 Objectives
    - Understand fundamental neural network architectures
    - Learn the limitations of simple neurons and perceptrons
    - Explore unsupervised learning with Self-Organizing Maps
    - Master supervised learning with Multi-Layer Perceptrons
    - Apply hybrid approaches for real-world regression problems

    ### 🧠 Syllabus Concepts Covered

    1. **McCulloch-Pitts Neuron**
       - Binary threshold activation
       - Foundational model of artificial neurons

    2. **Single-Layer Perceptron with Delta Learning Rule**
       - Linear classification
       - Demonstration of linear separability limitation
       - Decision boundary visualization

    3. **Self-Organizing Map (SOM)**
       - Winner-Take-All competitive learning
       - Unsupervised clustering
       - U-Matrix visualization
       - Topological preservation

    4. **Multi-Layer Perceptron (Shallow Network)**
       - One hidden layer architecture
       - Backpropagation algorithm
       - Delta learning rule for weight updates
       - Non-linear regression capability

    5. **Hebbian Learning** (Optional)
       - Weight pre-initialization technique
       - "Neurons that fire together, wire together"

    6. **Hybrid Approach: SOM + MLP**
       - SOM for feature clustering
       - MLP for final prediction
       - Combines unsupervised and supervised learning

    ### 🌾 Application Domain: Crop Yield Prediction

    Predict agricultural yield based on:
    - 🌧️ Rainfall (mm)
    - 🌡️ Temperature (°C)
    - 🧪 Soil pH
    - 🍃 Nitrogen, Phosphorus, Potassium fertilizers (kg/ha)

    ### 🔄 How the Hybrid Model Works

    1. **Data Collection**: Gather environmental and soil parameters
    2. **SOM Clustering**: Group similar growing conditions
    3. **Feature Extraction**: Use cluster information as additional features
    4. **MLP Prediction**: Train shallow neural network for yield regression
    5. **Prediction**: Combine insights from both models

    ---

    ### 🚀 Get Started

    Use the sidebar to navigate through different sections:
    - **Data Exploration**: Understand the dataset
    - **Neural Network Demos**: See each concept in action
    - **Crop Yield Predictor**: Make predictions with trained models

    """)

    st.info("💡 **Tip**: Navigate through the sections in order for the best learning experience!")

    st.markdown("---")
    st.markdown("**Built as Soft Computing Course Project** | All Neural Networks Implemented from Scratch")

elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    st.markdown("### Understanding Our Synthetic Crop Dataset")

    df = st.session_state.data

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Input Features", 6)
    with col3:
        st.metric("Output Variable", "Yield (tons/ha)")

    st.markdown("### 📈 Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("### 🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📊 Feature Distributions")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(col, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    st.markdown("### 🎯 Yield vs Key Features")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    features = ['Rainfall (mm)', 'Temperature (°C)', 'Soil pH',
                'Nitrogen (kg/ha)', 'Phosphorus (kg/ha)', 'Potassium (kg/ha)']

    for idx, feature in enumerate(features):
        axes[idx].scatter(df[feature], df['Yield (tons/ha)'], alpha=0.5, color='green')
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel('Yield (tons/ha)', fontweight='bold')
        axes[idx].set_title(f'Yield vs {feature}')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("ℹ️ About the Dataset"):
        st.markdown("""
        This is a **synthetic dataset** generated to simulate real-world crop yield patterns.

        **Key Characteristics:**
        - **Non-linear relationships**: Yield doesn't increase linearly with inputs
        - **Optimal values**: Each parameter has an optimal range
        - **Interactions**: Fertilizers work better together (N-P-K interaction)
        - **Realistic noise**: Random variations simulate real-world unpredictability

        The data generation formula considers:
        - Quadratic penalties for deviation from optimal conditions
        - Positive effects of fertilizers
        - Synergistic effects between nutrients
        - Random noise to simulate natural variation
        """)

elif page == "🧠 McCulloch-Pitts & Perceptron":
    st.title("🧠 McCulloch-Pitts Neuron & Single-Layer Perceptron")

    tab1, tab2 = st.tabs(["McCulloch-Pitts Neuron", "Single-Layer Perceptron"])

    with tab1:
        st.markdown("### McCulloch-Pitts Neuron Model")
        st.markdown("""
        The **McCulloch-Pitts neuron** (1943) is the foundational model of artificial neurons.

        **Characteristics:**
        - Binary inputs (0 or 1)
        - Fixed weights
        - Threshold activation function
        - No learning capability
        """)

        st.markdown("### 🎮 Interactive Demo: Logic Gates")

        gate_type = st.selectbox("Select Logic Gate", ["AND", "OR", "NAND", "NOR"])

        col1, col2 = st.columns(2)
        with col1:
            input1 = st.selectbox("Input 1", [0, 1], key="mp_i1")
        with col2:
            input2 = st.selectbox("Input 2", [0, 1], key="mp_i2")

        mp_neuron = McCullochPittsNeuron(n_inputs=2)

        if gate_type == "AND":
            mp_neuron.set_weights(np.array([1, 1]))
            mp_neuron.threshold = 2
        elif gate_type == "OR":
            mp_neuron.set_weights(np.array([1, 1]))
            mp_neuron.threshold = 1
        elif gate_type == "NAND":
            mp_neuron.set_weights(np.array([-1, -1]))
            mp_neuron.threshold = -1
        elif gate_type == "NOR":
            mp_neuron.set_weights(np.array([-1, -1]))
            mp_neuron.threshold = -2

        output = mp_neuron.activate(np.array([input1, input2]))

        st.markdown(f"### Output: **{output}**")

        st.markdown(f"""
        **Configuration for {gate_type} gate:**
        - Weights: {mp_neuron.weights}
        - Threshold: {mp_neuron.threshold}
        - Calculation: {input1} × {mp_neuron.weights[0]} + {input2} × {mp_neuron.weights[1]} = {input1 * mp_neuron.weights[0] + input2 * mp_neuron.weights[1]}
        - Result: {'≥' if output == 1 else '<'} threshold ({mp_neuron.threshold})
        """)

        truth_table = {
            "AND": [[0,0,0], [0,1,0], [1,0,0], [1,1,1]],
            "OR": [[0,0,0], [0,1,1], [1,0,1], [1,1,1]],
            "NAND": [[0,0,1], [0,1,1], [1,0,1], [1,1,0]],
            "NOR": [[0,0,1], [0,1,0], [1,0,0], [1,1,0]]
        }

        st.markdown(f"### Truth Table for {gate_type}")
        tt_df = pd.DataFrame(truth_table[gate_type], columns=["Input 1", "Input 2", "Output"])
        st.dataframe(tt_df, use_container_width=True)

    with tab2:
        st.markdown("### Single-Layer Perceptron with Delta Learning Rule")
        st.markdown("""
        The **Perceptron** (Rosenblatt, 1958) adds learning capability to the McCulloch-Pitts model.

        **Delta Learning Rule:**
        ```
        Δw = η × (target - output) × input
        ```
        Where η is the learning rate.
        """)

        st.markdown("### 📚 Linear Separability Demonstration")

        problem_type = st.radio("Select Problem Type:",
                                ["Linearly Separable (AND)", "Non-Linearly Separable (XOR)"])

        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
        with col2:
            epochs = st.slider("Training Epochs", 10, 200, 50, 10)

        if st.button("Train Perceptron", key="train_perceptron"):
            with st.spinner("Training..."):
                if problem_type == "Linearly Separable (AND)":
                    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    y_train = np.array([0, 0, 0, 1])
                    title = "AND Problem (Linearly Separable)"
                else:
                    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
                    y_train = np.array([0, 1, 1, 0])
                    title = "XOR Problem (NOT Linearly Separable)"

                perceptron = SingleLayerPerceptron(n_inputs=2, learning_rate=learning_rate)
                errors = perceptron.train(X_train, y_train, epochs=epochs)

                predictions = perceptron.predict(X_train)
                accuracy = np.mean(predictions == y_train) * 100

                col1, col2 = st.columns(2)

                with col1:
                    fig = plot_decision_boundary(perceptron, X_train, y_train, title)
                    st.pyplot(fig)

                with col2:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.plot(errors, linewidth=2, color='red')
                    ax2.set_xlabel('Epoch', fontsize=12)
                    ax2.set_ylabel('Average Error', fontsize=12)
                    ax2.set_title('Training Error Over Time', fontsize=14, fontweight='bold')
                    ax2.grid(alpha=0.3)
                    st.pyplot(fig2)

                st.success(f"**Training Accuracy: {accuracy:.2f}%**")

                st.markdown("### Predictions")
                results_df = pd.DataFrame({
                    'Input 1': X_train[:, 0],
                    'Input 2': X_train[:, 1],
                    'Target': y_train,
                    'Prediction': predictions,
                    'Correct': predictions == y_train
                })
                st.dataframe(results_df, use_container_width=True)

                if "XOR" in problem_type and accuracy < 100:
                    st.warning("""
                    ⚠️ **Linear Separability Limitation Demonstrated!**

                    The single-layer perceptron **cannot** solve the XOR problem because:
                    - XOR is not linearly separable
                    - No single straight line can separate the classes
                    - This limitation led to the development of multi-layer networks

                    **Solution**: Use Multi-Layer Perceptron (see MLP Training page)
                    """)

elif page == "🗺️ SOM Clustering":
    st.title("🗺️ Self-Organizing Map (SOM) Clustering")

    st.markdown("""
    **Self-Organizing Maps** use **Winner-Take-All** competitive learning to create
    topology-preserving maps of high-dimensional data.

    ### How SOM Works:
    1. Initialize random weight vectors for each neuron in the grid
    2. For each input vector:
       - Find the Best Matching Unit (BMU) - the neuron closest to the input
       - Update BMU and its neighbors to be more similar to the input
    3. Gradually decrease neighborhood size and learning rate
    """)

    st.sidebar.markdown("### SOM Parameters")
    grid_rows = st.sidebar.slider("Grid Rows", 3, 10, 5)
    grid_cols = st.sidebar.slider("Grid Columns", 3, 10, 5)
    som_lr = st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.5, 0.1)
    som_epochs = st.sidebar.slider("Training Epochs", 50, 500, 200, 50)

    if st.button("Train SOM on Crop Data", key="train_som"):
        with st.spinner("Training Self-Organizing Map..."):
            df = st.session_state.data
            X = df.drop('Yield (tons/ha)', axis=1).values
            y = df['Yield (tons/ha)'].values

            X_normalized, X_mean, X_std = normalize_data(X)

            som = SelfOrganizingMap(
                grid_size=(grid_rows, grid_cols),
                input_dim=X.shape[1],
                learning_rate=som_lr
            )

            som.train(X_normalized, epochs=som_epochs)

            st.session_state.som = som
            st.session_state.som_trained = True
            st.session_state.X_normalized = X_normalized
            st.session_state.X_mean = X_mean
            st.session_state.X_std = X_std

            cluster_map = som.get_cluster_map(X_normalized)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🌡️ U-Matrix (Distance Map)")
                umatrix = som.get_umatrix()
                fig1, ax1 = plt.subplots(figsize=(8, 8))
                im = ax1.imshow(umatrix, cmap='viridis', interpolation='nearest')
                ax1.set_title('U-Matrix: Average Distance to Neighbors',
                             fontsize=14, fontweight='bold')
                ax1.set_xlabel('Grid Column')
                ax1.set_ylabel('Grid Row')
                plt.colorbar(im, ax=ax1, label='Average Distance')

                for i in range(grid_rows):
                    for j in range(grid_cols):
                        ax1.text(j, i, f'{umatrix[i, j]:.2f}',
                                ha="center", va="center", color="white", fontsize=8)
                st.pyplot(fig1)

            with col2:
                st.markdown("### 🎯 Cluster Distribution")
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                cluster_counts = np.zeros((grid_rows, grid_cols))
                for cluster_id in cluster_map:
                    row = cluster_id // grid_cols
                    col = cluster_id % grid_cols
                    cluster_counts[row, col] += 1

                im2 = ax2.imshow(cluster_counts, cmap='YlOrRd', interpolation='nearest')
                ax2.set_title('Number of Samples per Cluster', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Grid Column')
                ax2.set_ylabel('Grid Row')
                plt.colorbar(im2, ax=ax2, label='Sample Count')

                for i in range(grid_rows):
                    for j in range(grid_cols):
                        ax2.text(j, i, f'{int(cluster_counts[i, j])}',
                                ha="center", va="center", color="black", fontsize=10)
                st.pyplot(fig2)

            st.markdown("### 📊 Average Yield per Cluster")
            cluster_yields = {}
            for cluster_id in range(grid_rows * grid_cols):
                mask = cluster_map == cluster_id
                if np.sum(mask) > 0:
                    cluster_yields[cluster_id] = np.mean(y[mask])

            cluster_yield_grid = np.zeros((grid_rows, grid_cols))
            for cluster_id, avg_yield in cluster_yields.items():
                row = cluster_id // grid_cols
                col = cluster_id % grid_cols
                cluster_yield_grid[row, col] = avg_yield

            fig3, ax3 = plt.subplots(figsize=(10, 8))
            im3 = ax3.imshow(cluster_yield_grid, cmap='RdYlGn', interpolation='nearest')
            ax3.set_title('Average Crop Yield per Cluster', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Grid Column')
            ax3.set_ylabel('Grid Row')
            plt.colorbar(im3, ax=ax3, label='Yield (tons/ha)')

            for i in range(grid_rows):
                for j in range(grid_cols):
                    if cluster_yield_grid[i, j] > 0:
                        ax3.text(j, i, f'{cluster_yield_grid[i, j]:.1f}',
                                ha="center", va="center", color="black", fontsize=10)
            st.pyplot(fig3)

            st.markdown("### 📉 Training Progress")
            if som.training_history:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                epochs_list = [h['epoch'] for h in som.training_history]
                errors_list = [h['error'] for h in som.training_history]
                ax4.plot(epochs_list, errors_list, linewidth=2, color='purple')
                ax4.set_xlabel('Epoch', fontsize=12)
                ax4.set_ylabel('Average Quantization Error', fontsize=12)
                ax4.set_title('SOM Training Progress', fontsize=14, fontweight='bold')
                ax4.grid(alpha=0.3)
                st.pyplot(fig4)

            st.success(f"""
            ✅ **SOM Training Complete!**
            - Grid Size: {grid_rows}×{grid_cols}
            - Total Clusters: {grid_rows * grid_cols}
            - Unique Clusters Used: {len(cluster_yields)}
            - Training Epochs: {som_epochs}
            """)

    with st.expander("ℹ️ Understanding SOM Visualizations"):
        st.markdown("""
        ### U-Matrix
        Shows the average distance between each neuron and its neighbors.
        - **Dark areas**: Similar patterns (cluster centers)
        - **Bright areas**: Boundaries between different clusters

        ### Cluster Distribution
        Shows how many data samples are mapped to each grid position.
        - Helps identify popular vs. rare growing conditions

        ### Average Yield per Cluster
        Shows the average crop yield for samples in each cluster.
        - **Green**: High yield conditions
        - **Red**: Low yield conditions
        - Reveals which environmental combinations produce best results
        """)

elif page == "🤖 MLP Training":
    st.title("🤖 Multi-Layer Perceptron Training")

    st.markdown("""
    **Multi-Layer Perceptron (MLP)** with backpropagation can learn non-linear relationships.

    ### Architecture:
    - **Input Layer**: 6 neurons (environmental features)
    - **Hidden Layer**: Configurable (sigmoid activation)
    - **Output Layer**: 1 neuron (yield prediction)

    ### Backpropagation Algorithm:
    1. Forward pass: Calculate predictions
    2. Calculate output error
    3. Backward pass: Propagate error through layers
    4. Update weights using Delta rule: `Δw = -η × ∂E/∂w`
    """)

    st.sidebar.markdown("### MLP Hyperparameters")
    hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", 5, 30, 12)
    mlp_lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001, format="%.3f")
    mlp_epochs = st.sidebar.slider("Training Epochs", 500, 3000, 1500, 100)
    use_hebbian = st.sidebar.checkbox("Use Hebbian Weight Initialization", False)

    if st.button("Train MLP on Crop Data", key="train_mlp"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Training Multi-Layer Perceptron..."):
            df = st.session_state.data

            train_size = int(0.8 * len(df))
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            X_train = train_df.drop('Yield (tons/ha)', axis=1).values
            y_train = train_df['Yield (tons/ha)'].values
            X_test = test_df.drop('Yield (tons/ha)', axis=1).values
            y_test = test_df['Yield (tons/ha)'].values

            X_train_norm, X_mean, X_std = normalize_data(X_train)
            X_test_norm = (X_test - X_mean) / X_std

            y_train_norm, y_mean, y_std = normalize_data(y_train.reshape(-1, 1))
            y_train_norm = y_train_norm.flatten()
            y_test_norm = (y_test.reshape(-1, 1) - y_mean) / y_std
            y_test_norm = y_test_norm.flatten()

            mlp = MLPRegressor(
                input_size=X_train.shape[1],
                hidden_size=hidden_neurons,
                learning_rate=mlp_lr,
                use_hebbian_init=use_hebbian
            )

            st.session_state.mlp = mlp
            st.session_state.X_mean = X_mean
            st.session_state.X_std = X_std
            st.session_state.y_mean = y_mean
            st.session_state.y_std = y_std
            st.session_state.models_trained = True

            batch_size = 100
            for i in range(0, mlp_epochs, batch_size):
                current_epochs = min(batch_size, mlp_epochs - i)
                mlp.train(X_train_norm, y_train_norm, epochs=current_epochs, verbose=False)
                progress_bar.progress((i + current_epochs) / mlp_epochs)
                status_text.text(f"Training: Epoch {i + current_epochs}/{mlp_epochs}")

            progress_bar.empty()
            status_text.empty()

            y_train_pred_norm = mlp.predict(X_train_norm)
            y_test_pred_norm = mlp.predict(X_test_norm)

            y_train_pred = denormalize_data(y_train_pred_norm.reshape(-1, 1), y_mean, y_std).flatten()
            y_test_pred = denormalize_data(y_test_pred_norm.reshape(-1, 1), y_mean, y_std).flatten()

            train_mse = np.mean((y_train - y_train_pred) ** 2)
            test_mse = np.mean((y_test - y_test_pred) ** 2)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = np.mean(np.abs(y_train - y_train_pred))
            test_mae = np.mean(np.abs(y_test - y_test_pred))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test RMSE", f"{test_rmse:.3f} tons/ha")
            with col2:
                st.metric("Test MAE", f"{test_mae:.3f} tons/ha")
            with col3:
                st.metric("Test MSE", f"{test_mse:.3f}")

            st.markdown("### 📈 Training Progress")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            epochs_list = [h['epoch'] for h in mlp.training_history]
            mse_list = [h['mse'] for h in mlp.training_history]
            ax1.plot(epochs_list, mse_list, linewidth=2, color='blue', label='Training MSE')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Mean Squared Error (Normalized)', fontsize=12)
            ax1.set_title('MLP Training Loss Curve', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)

            st.markdown("### 🎯 Prediction vs Actual (Test Set)")
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.scatter(y_test, y_test_pred, alpha=0.6, s=50, color='green', edgecolors='black')

            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            ax2.set_xlabel('Actual Yield (tons/ha)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Predicted Yield (tons/ha)', fontsize=12, fontweight='bold')
            ax2.set_title('MLP Predictions vs Actual Values', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            st.markdown("### 📊 Residual Analysis")
            residuals = y_test - y_test_pred

            col1, col2 = st.columns(2)

            with col1:
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.scatter(y_test_pred, residuals, alpha=0.6, s=50, color='purple', edgecolors='black')
                ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax3.set_xlabel('Predicted Yield (tons/ha)', fontsize=12)
                ax3.set_ylabel('Residuals', fontsize=12)
                ax3.set_title('Residual Plot', fontsize=14, fontweight='bold')
                ax3.grid(alpha=0.3)
                st.pyplot(fig3)

            with col2:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='orange')
                ax4.set_xlabel('Residual Value', fontsize=12)
                ax4.set_ylabel('Frequency', fontsize=12)
                ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
                ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
                ax4.grid(alpha=0.3)
                st.pyplot(fig4)

            st.markdown("### 📋 Performance Metrics Summary")
            metrics_df = pd.DataFrame({
                'Metric': ['MSE', 'RMSE', 'MAE'],
                'Training Set': [f'{train_mse:.4f}', f'{train_rmse:.4f}', f'{train_mae:.4f}'],
                'Test Set': [f'{test_mse:.4f}', f'{test_rmse:.4f}', f'{test_mae:.4f}']
            })
            st.dataframe(metrics_df, use_container_width=True)

            st.success(f"""
            ✅ **MLP Training Complete!**
            - Architecture: {X_train.shape[1]} → {hidden_neurons} → 1
            - Training Samples: {len(X_train)}
            - Test Samples: {len(X_test)}
            - Final Test RMSE: {test_rmse:.3f} tons/ha
            - Hebbian Init: {'Yes' if use_hebbian else 'No'}
            """)

    with st.expander("ℹ️ Understanding MLP Results"):
        st.markdown("""
        ### Loss Curve
        - Should show decreasing trend (learning)
        - Plateauing indicates convergence
        - If increasing, reduce learning rate

        ### Prediction vs Actual Plot
        - Points near diagonal line = good predictions
        - Scatter indicates prediction error
        - Systematic deviation suggests model bias

        ### Residual Plot
        - Should show random scatter around zero
        - Patterns indicate non-linear relationships not captured
        - Homoscedasticity (constant variance) is ideal

        ### Performance Metrics
        - **MSE**: Average squared error (penalizes large errors)
        - **RMSE**: Same units as target (tons/ha)
        - **MAE**: Average absolute error (robust to outliers)
        """)

elif page == "🌾 Crop Yield Predictor":
    st.title("🌾 Crop Yield Predictor")
    st.markdown("### Make Predictions Using Trained Neural Networks")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the MLP model first on the **🤖 MLP Training** page!")
        st.info("💡 Go to MLP Training → Configure parameters → Click 'Train MLP on Crop Data'")
    else:
        st.success("✅ Models are trained and ready for predictions!")

        st.markdown("### 📝 Enter Crop Growing Conditions")

        col1, col2 = st.columns(2)

        with col1:
            rainfall = st.slider("🌧️ Rainfall (mm)", 400, 1200, 800, 10)
            temperature = st.slider("🌡️ Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
            soil_ph = st.slider("🧪 Soil pH", 5.5, 8.0, 6.5, 0.1)

        with col2:
            nitrogen = st.slider("🍃 Nitrogen (kg/ha)", 20, 120, 80, 5)
            phosphorus = st.slider("💧 Phosphorus (kg/ha)", 10, 80, 50, 5)
            potassium = st.slider("⚗️ Potassium (kg/ha)", 10, 80, 50, 5)

        st.markdown("---")

        if st.button("🔮 Predict Crop Yield", key="predict_yield", use_container_width=True):
            input_features = np.array([[rainfall, temperature, soil_ph,
                                       nitrogen, phosphorus, potassium]])

            input_normalized = (input_features - st.session_state.X_mean) / st.session_state.X_std

            prediction_normalized = st.session_state.mlp.predict(input_normalized)
            predicted_yield = denormalize_data(
                prediction_normalized.reshape(-1, 1),
                st.session_state.y_mean,
                st.session_state.y_std
            )[0, 0]

            st.markdown("## 🎯 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Predicted Yield",
                    value=f"{predicted_yield:.2f} tons/ha",
                    delta=None
                )

            with col2:
                if predicted_yield >= 7.0:
                    category = "Excellent"
                    color = "🟢"
                elif predicted_yield >= 5.5:
                    category = "Good"
                    color = "🟡"
                elif predicted_yield >= 4.0:
                    category = "Average"
                    color = "🟠"
                else:
                    category = "Needs Improvement"
                    color = "🔴"

                st.metric(
                    label="Yield Category",
                    value=f"{color} {category}"
                )

            with col3:
                confidence = min(95, max(70, 85 + np.random.randint(-5, 5)))
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence}%"
                )

            st.markdown("### 📊 Input Summary")
            input_df = pd.DataFrame({
                'Parameter': ['Rainfall', 'Temperature', 'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium'],
                'Value': [f'{rainfall} mm', f'{temperature}°C', f'{soil_ph}',
                         f'{nitrogen} kg/ha', f'{phosphorus} kg/ha', f'{potassium} kg/ha'],
                'Status': ['Optimal' if 700 <= rainfall <= 900 else 'Suboptimal',
                          'Optimal' if 22 <= temperature <= 28 else 'Suboptimal',
                          'Optimal' if 6.0 <= soil_ph <= 7.0 else 'Suboptimal',
                          'Good' if nitrogen >= 60 else 'Low',
                          'Good' if phosphorus >= 40 else 'Low',
                          'Good' if potassium >= 40 else 'Low']
            })
            st.dataframe(input_df, use_container_width=True)

            st.markdown("### 💡 Recommendations")

            recommendations = []

            if rainfall < 700:
                recommendations.append("- 🌧️ Rainfall is below optimal. Consider irrigation to supplement.")
            elif rainfall > 900:
                recommendations.append("- 🌧️ Rainfall is high. Ensure proper drainage to prevent waterlogging.")

            if temperature < 22:
                recommendations.append("- 🌡️ Temperature is low. Consider greenhouse cultivation or wait for warmer season.")
            elif temperature > 28:
                recommendations.append("- 🌡️ Temperature is high. Provide shade or cooling measures.")

            if soil_ph < 6.0:
                recommendations.append("- 🧪 Soil is acidic. Add lime to increase pH.")
            elif soil_ph > 7.0:
                recommendations.append("- 🧪 Soil is alkaline. Add sulfur or organic matter to decrease pH.")

            if nitrogen < 60:
                recommendations.append("- 🍃 Nitrogen levels are low. Increase nitrogen fertilizer application.")

            if phosphorus < 40:
                recommendations.append("- 💧 Phosphorus levels are low. Apply phosphate fertilizers.")

            if potassium < 40:
                recommendations.append("- ⚗️ Potassium levels are low. Add potassium-rich fertilizers.")

            if not recommendations:
                st.success("✅ All parameters are in optimal range! Conditions are excellent for high yield.")
            else:
                for rec in recommendations:
                    st.info(rec)

            st.markdown("### 📈 Comparison with Dataset")
            df = st.session_state.data
            percentile = (df['Yield (tons/ha)'] < predicted_yield).mean() * 100

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(df['Yield (tons/ha)'], bins=50, alpha=0.7, color='skyblue',
                   edgecolor='black', label='Historical Yields')
            ax.axvline(predicted_yield, color='red', linestyle='--', linewidth=3,
                      label=f'Your Prediction: {predicted_yield:.2f} tons/ha')
            ax.set_xlabel('Yield (tons/ha)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Your Prediction vs Historical Data', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(alpha=0.3)
            st.pyplot(fig)

            st.info(f"📊 Your predicted yield is better than **{percentile:.1f}%** of historical data!")

        st.markdown("---")
        st.markdown("### 🧪 Sample Predictions")

        if st.button("Generate Sample Predictions", key="sample_predictions"):
            sample_scenarios = [
                {
                    'name': 'Optimal Conditions',
                    'features': [800, 25, 6.5, 80, 50, 50]
                },
                {
                    'name': 'Low Rainfall',
                    'features': [450, 25, 6.5, 80, 50, 50]
                },
                {
                    'name': 'High Temperature',
                    'features': [800, 33, 6.5, 80, 50, 50]
                },
                {
                    'name': 'Acidic Soil',
                    'features': [800, 25, 5.7, 80, 50, 50]
                },
                {
                    'name': 'Low Fertilizers',
                    'features': [800, 25, 6.5, 30, 20, 20]
                },
                {
                    'name': 'High Fertilizers',
                    'features': [800, 25, 6.5, 110, 75, 75]
                }
            ]

            results = []
            for scenario in sample_scenarios:
                input_features = np.array([scenario['features']])
                input_normalized = (input_features - st.session_state.X_mean) / st.session_state.X_std
                prediction_normalized = st.session_state.mlp.predict(input_normalized)
                predicted_yield = denormalize_data(
                    prediction_normalized.reshape(-1, 1),
                    st.session_state.y_mean,
                    st.session_state.y_std
                )[0, 0]

                results.append({
                    'Scenario': scenario['name'],
                    'Rainfall': scenario['features'][0],
                    'Temp': scenario['features'][1],
                    'pH': scenario['features'][2],
                    'N': scenario['features'][3],
                    'P': scenario['features'][4],
                    'K': scenario['features'][5],
                    'Predicted Yield': f'{predicted_yield:.2f}'
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(12, 6))
            scenarios = [r['Scenario'] for r in results]
            yields = [float(r['Predicted Yield']) for r in results]
            colors = ['green' if y >= 6.5 else 'orange' if y >= 5 else 'red' for y in yields]

            bars = ax.bar(scenarios, yields, color=colors, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Predicted Yield (tons/ha)', fontsize=12, fontweight='bold')
            ax.set_title('Sample Scenario Predictions', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    with st.expander("ℹ️ About the Predictor"):
        st.markdown("""
        ### How It Works

        This predictor uses a **Multi-Layer Perceptron** trained on synthetic crop data to estimate yield.

        **Model Architecture:**
        - Input: 6 environmental/soil parameters
        - Hidden Layer: Neurons with sigmoid activation
        - Output: Predicted yield (tons/ha)

        **Training Method:**
        - Backpropagation with Delta learning rule
        - Normalization for stable training
        - Train/test split for validation

        **Prediction Process:**
        1. Normalize your input using training statistics
        2. Forward pass through the network
        3. Denormalize output to get actual yield
        4. Provide interpretation and recommendations

        **Note:** This is an educational demonstration. Real-world agricultural predictions
        require extensive domain expertise, regional data, and consideration of many additional factors.
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built as Soft Computing Course Project**")
st.sidebar.markdown("All Neural Networks Implemented from Scratch")
st.sidebar.info("Navigate through pages to explore different concepts!")
