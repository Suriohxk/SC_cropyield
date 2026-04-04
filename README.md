# 🌾 Crop Yield Predictor Using Shallow Neural Networks

A complete Streamlit web application demonstrating Soft Computing concepts through agricultural yield prediction.

## 📚 Project Overview

This educational application implements various neural network architectures **from scratch** using only NumPy to predict crop yields based on environmental and soil parameters.

### Concepts Covered

1. **McCulloch-Pitts Neuron** - Binary threshold neurons implementing logic gates
2. **Single-Layer Perceptron** - Delta learning rule with linear separability demonstration
3. **Self-Organizing Map (SOM)** - Winner-Take-All competitive learning with U-Matrix visualization
4. **Multi-Layer Perceptron (MLP)** - Backpropagation with shallow architecture (1 hidden layer)
5. **Hebbian Learning** - Optional weight pre-initialization
6. **Hybrid Approach** - Combining SOM clustering with MLP regression

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this project**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📱 Application Structure

The app consists of 6 main pages:

### 1. 🏠 Home
- Project overview
- Syllabus concepts explanation
- Learning objectives

### 2. 📊 Data Exploration
- Synthetic crop dataset (500 samples)
- Statistical analysis
- Correlation heatmap
- Feature distributions

### 3. 🧠 McCulloch-Pitts & Perceptron
- **McCulloch-Pitts**: Interactive logic gate demonstrations
- **Single-Layer Perceptron**: Training with Delta rule
- **Linear Separability**: Visual demonstration of XOR problem limitation
- Decision boundary visualization

### 4. 🗺️ SOM Clustering
- Self-Organizing Map training
- U-Matrix visualization
- Cluster distribution map
- Average yield per cluster analysis

### 5. 🤖 MLP Training
- Multi-Layer Perceptron with backpropagation
- Configurable hyperparameters
- Training loss curves
- Performance metrics (MSE, RMSE, MAE)
- Residual analysis

### 6. 🌾 Crop Yield Predictor
- Interactive prediction interface
- Input sliders for all parameters
- Yield prediction with interpretation
- Personalized recommendations
- Sample scenario comparisons

## 🎮 How to Use

1. **Start at Home** - Understand the project objectives

2. **Explore Data** - Familiarize yourself with the dataset

3. **Learn Concepts** - Visit McCulloch-Pitts & Perceptron page
   - Try different logic gates
   - Train perceptron on AND and XOR problems
   - Observe linear separability limitation

4. **Train SOM** - Go to SOM Clustering page
   - Adjust grid size and learning parameters
   - Click "Train SOM on Crop Data"
   - Analyze U-Matrix and cluster patterns

5. **Train MLP** - Navigate to MLP Training page
   - Configure hidden neurons and learning rate
   - Click "Train MLP on Crop Data"
   - Review training curves and metrics

6. **Make Predictions** - Use the Crop Yield Predictor
   - Adjust environmental parameters with sliders
   - Click "Predict Crop Yield"
   - Get predictions and recommendations

## 🧪 Features

### Educational Features
- All neural networks implemented from scratch (no Keras/PyTorch)
- Step-by-step demonstrations of each concept
- Interactive parameter tuning
- Visual explanations and tooltips
- Real-time training progress

### Technical Features
- Synthetic data generation (no external datasets required)
- Proper train/test split
- Data normalization
- Multiple evaluation metrics
- Beautiful visualizations using Matplotlib and Seaborn

### User Experience
- Clean, modern UI
- Sidebar navigation
- Progress indicators
- Interactive widgets
- Responsive visualizations

## 📊 Model Performance

Typical performance metrics on test set:
- **RMSE**: 0.3-0.5 tons/ha
- **MAE**: 0.2-0.4 tons/ha
- **Accuracy**: ~85-95% within 0.5 tons/ha

## 🎓 Learning Outcomes

After using this application, you will understand:

1. How McCulloch-Pitts neurons form the foundation of neural networks
2. Why single-layer perceptrons cannot solve non-linearly separable problems
3. How Self-Organizing Maps create topology-preserving representations
4. How backpropagation trains multi-layer networks
5. The difference between supervised and unsupervised learning
6. Practical application of neural networks to regression problems

## 🔧 Customization

### Adjustable Hyperparameters

**Single-Layer Perceptron:**
- Learning rate: 0.01 - 1.0
- Training epochs: 10 - 200

**Self-Organizing Map:**
- Grid size: 3×3 to 10×10
- Learning rate: 0.1 - 1.0
- Training epochs: 50 - 500

**Multi-Layer Perceptron:**
- Hidden neurons: 5 - 30
- Learning rate: 0.001 - 0.1
- Training epochs: 500 - 3000
- Hebbian initialization: On/Off

### Data Generation

The synthetic dataset simulates realistic crop yield patterns based on:
- Rainfall (400-1200 mm)
- Temperature (15-35°C)
- Soil pH (5.5-8.0)
- Nitrogen fertilizer (20-120 kg/ha)
- Phosphorus fertilizer (10-80 kg/ha)
- Potassium fertilizer (10-80 kg/ha)

## 📝 Code Structure

```
app.py                      # Main application file
├── McCullochPittsNeuron   # Class for M-P neuron
├── SingleLayerPerceptron  # Perceptron with Delta rule
├── SelfOrganizingMap      # SOM with Winner-Take-All
├── MLPRegressor           # MLP with backpropagation
├── generate_crop_data()   # Synthetic data generator
├── normalize_data()       # Data normalization utilities
└── Streamlit pages        # Multi-page application
```

## 🐛 Troubleshooting

**Issue**: Models not trained error on Predictor page
- **Solution**: Navigate to "🤖 MLP Training" page and click "Train MLP on Crop Data"

**Issue**: Slow training
- **Solution**: Reduce number of epochs or use fewer hidden neurons

**Issue**: Poor predictions
- **Solution**: Increase hidden neurons, adjust learning rate, or train for more epochs

## 📚 References

- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage
- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors

## 👨‍🎓 Course Information

**Project**: Crop Yield Predictor Using Shallow Neural Networks
**Course**: Soft Computing
**Implementation**: From scratch using NumPy
**Framework**: Streamlit

---

**Built as Soft Computing Course Project**
All Neural Networks Implemented from Scratch

## 📄 License

This is an educational project for academic purposes.
