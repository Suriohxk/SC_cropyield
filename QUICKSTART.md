# 🚀 Quick Start Guide

Get the Crop Yield Predictor running in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter an "externally-managed-environment" error, create a virtual environment first:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run the Application

```bash
streamlit run app.py
```

## Step 3: Open in Browser

The app will automatically open at: **http://localhost:8501**

If it doesn't open automatically, manually navigate to the URL in your browser.

## 🎯 First Time Usage

1. **Start at Home** (🏠) - Read the project overview
2. **Explore Data** (📊) - Understand the dataset
3. **Learn Neural Networks** (🧠) - Try McCulloch-Pitts & Perceptron demos
4. **Train SOM** (🗺️) - Click "Train SOM on Crop Data"
5. **Train MLP** (🤖) - Click "Train MLP on Crop Data"
6. **Make Predictions** (🌾) - Use the interactive predictor

## ⚙️ Recommended Settings for First Run

**SOM Training:**
- Grid: 5×5
- Learning Rate: 0.5
- Epochs: 200

**MLP Training:**
- Hidden Neurons: 12
- Learning Rate: 0.01
- Epochs: 1500

## 💡 Tips

- Training takes 10-30 seconds depending on epochs
- You can retrain models with different parameters anytime
- All data is synthetic - no external files needed
- Use sidebar to navigate between pages

## 🆘 Troubleshooting

**"Please train the MLP model first" error:**
- Go to "🤖 MLP Training" page
- Click "Train MLP on Crop Data" button
- Wait for training to complete

**Slow performance:**
- Reduce epochs (try 1000 instead of 1500)
- Reduce hidden neurons (try 8 instead of 12)

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

## 🎓 Learning Path

Follow this sequence for best understanding:

1. McCulloch-Pitts Neuron → Understand basic neuron model
2. Single-Layer Perceptron → Learn about linear separability
3. See XOR failure → Understand need for multiple layers
4. SOM Clustering → Learn unsupervised learning
5. MLP Training → Master supervised learning
6. Make Predictions → Apply knowledge

---

**Happy Learning!** 🌾🧠
