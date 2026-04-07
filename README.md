# 🌾 Crop Yield Predictor Using Shallow Neural Networks

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

## App Sections

1. **Home** - Project overview
2. **Data + Perceptron** - Dataset exploration and perceptron demo (AND/XOR)
3. **SOM Clustering** - Self-Organizing Map training and U-Matrix visualization
4. **MLP + Crop Predictor** - MLP training and interactive yield prediction

## Features

- Single-Layer Perceptron (Delta rule)
- Self-Organizing Map (Winner-Take-All learning)
- Multi-Layer Perceptron (Backpropagation)
- Synthetic crop data (500 samples, 6 features)
- Interactive prediction with 6 input sliders
- Trained entirely with NumPy (no TensorFlow/Keras)

## Hyperparameters

**Perceptron:** Learning rate, epochs
**SOM:** Grid size, epochs
**MLP:** Hidden neurons, learning rate, epochs

## Expected Results

- Perceptron: AND (100%), XOR (50%)
- SOM: Clusters into grid, U-Matrix shows boundaries
- MLP: RMSE 0.3-0.5 tons/ha

## Files

- `app.py` - Main application (718 lines)
- `requirements.txt` - Dependencies
