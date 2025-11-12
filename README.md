# AI-Driven Crop Yield Prediction for Climate Action

## Introduction
This project addresses **UN SDG 13: Climate Action** by using a supervised learning neural network to predict crop yields based on climate variables (temperature, rainfall, pesticides). By forecasting yields under changing climate conditions, it helps farmers optimize resources, reduce emissions, and combat food insecurity (SDG 2). The model is built using PyTorch and demonstrates Week 2 concepts like neural networks and supervised learning.

## Problem Statement
Climate change disrupts agriculture through extreme weather, reducing yields by up to 20% for crops like wheat by 2050. Traditional forecasting is often inaccurate, leaving farmers unprepared. This AI solution provides data-driven predictions to support sustainable farming and reduce inequality in vulnerable regions.

## Solution Overview
- **Model**: A multi-layer perceptron (MLP) neural network for regression.
- **Inputs**: Temperature (°C), rainfall (mm/year), pesticides (tons).
- **Output**: Crop yield (tons/hectare).
- **Tech Stack**: Python, PyTorch, matplotlib (visualization), Google Colab.
- **Data**: Synthetic dataset mimicking real-world climate-yield data (e.g., Kaggle’s Crop Yield Prediction Dataset).

## Demo Screenshots
### 1. Training Output
Shows the model’s training progress with decreasing loss over 500 epochs.

![Training Output](images/training_output.png)

### 2. Yield Prediction
Example prediction for a sample input (e.g., 25°C, 1000mm rainfall, 20 tons pesticides).

![Prediction Output](images/prediction_output.png)

### 3. Prediction vs. Actual Yield Plot
Visualizes model performance by comparing predicted vs. actual yields.

![Yield Plot]<img width="800" height="600" alt="yield_plot" src="https://github.com/user-attachments/assets/308b38d8-e0c9-4410-ba10-627a78a6bca2" />


## How to Run
1. Clone this repo: `git clone <repo-url>`
2. Install dependencies: `pip install torch numpy matplotlib`
3. Run the script: `python yield_predictor.py`
4. Or, open `notebook.ipynb` in Google Colab for an interactive demo.

## Files
- `yield_predictor.py`: Main script with model training and prediction.
- `notebook.ipynb`: Colab-friendly version with visualizations.
- `images/`: Folder containing demo screenshots.

## Impact
This project aligns with the UN’s vision of AI as a bridge to sustainability. It empowers farmers with predictive insights, supports climate-resilient agriculture, and can scale globally with real-time climate data APIs.

## Future Work
- Integrate real datasets (e.g., FAO, World Bank).
- Add features like soil quality or CO2 levels.
- Deploy as a web app for farmer access.

## References
- UN SDG 13: Climate Action
- Kaggle Crop Yield Prediction Dataset

---

*“AI can be the bridge between innovation and sustainability.” — UN Tech Envoy*
