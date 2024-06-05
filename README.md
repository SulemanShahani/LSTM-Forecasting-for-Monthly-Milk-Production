# LSTM-Forecasting-for-Monthly-Milk-Production
This repository contains code to forecast monthly milk production using an LSTM (Long Short-Term Memory) neural network model. The dataset used in this project is monthly_milk_production.csv, which contains monthly milk production values from January 1962 to December 1975.

Steps
1. Data Loading and Visualization
The dataset monthly_milk_production.csv is loaded.
The data is visualized using matplotlib to understand its structure, trend, and seasonality.
2. Seasonal Decomposition
Seasonal decomposition is performed using seasonal_decompose from statsmodels.tsa.seasonal.
The decomposed series is plotted to visualize trend, seasonality, and residuals.
3. Data Preparation
The dataset is split into training and test sets.
Data scaling is applied using MinMaxScaler from sklearn.preprocessing.
4. Model Development
An LSTM neural network model is defined using keras.
Input/output sequences are generated using TimeseriesGenerator to train the model.
5. Model Training
The LSTM model is trained using the training dataset.
The model is evaluated using the test dataset.
Training progress and loss are monitored and visualized.
6. Model Evaluation and Forecasting
The trained model is used to forecast production values for the test set.
Forecasts are inverted from scaled values to the original data scale using scaler.inverse_transform.
Actual vs. predicted production values are plotted using matplotlib.
Root Mean Squared Error (RMSE) is calculated to evaluate the model's performance.
Requirements
Python 3.x
pandas
numpy
matplotlib
scikit-learn
statsmodels
keras (TensorFlow backend)
How to Use
Clone the repository:

git clone https://github.com/yourusername/milk-production-lstm.git
cd milk-production-lstm

Install the required packages:

pip install -r requirements.txt

Run the Jupyter Notebook or Python script:

Run milk_production_lstm.ipynb in Jupyter Notebook or Jupyter Lab.

Alternatively, run the Python script milk_production_lstm.py using:

python milk_production_lstm.py

View the results:

After running the notebook or script, you should see plots comparing actual vs. predicted milk production.
The RMSE value will also be printed, indicating the model's performance.
Files
monthly_milk_production.csv: Dataset containing monthly milk production values.
milk_production_lstm.ipynb: Jupyter Notebook containing the LSTM model implementation.
README.md: This file, containing information about the project, how to use the code, and other relevant details.
