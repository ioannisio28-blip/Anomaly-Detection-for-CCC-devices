import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


from sklearn.preprocessing import MinMaxScaler
import warnings

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K






class Evaluation:
    """" The dataframe that is used for the evaluation 
    should contain only the columns that have features. Any other unnesessary columns (like timestamp or device_id) 
    should be dropped before passing the dataframe to the Evaluation class."""


    def __init__(self, df: pd.DataFrame,time_steps: int = 200):
        self.model = None
        self.time_steps = time_steps
        self.df = df

        self.scaler = None
        self.X_real = None
        self.X_pred = None
        self.reconstruction_errors = None
        self.sorted_anomalies = None


    def load_a_model(self,path:str):
        self.model = load_model(path)

    def preprocess(self, df): # , columns_to_drop: list = None
        # Drop missing values
        df = df.dropna()

        # Drop specified columns
        # if columns_to_drop:
        #     features = df.drop(columns=columns_to_drop).copy()
        # else:
        #     features = df.copy()
        features = df.copy()

        # Normalize
        self.scaler = MinMaxScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Sliding window
        X_real = []
        for i in range(len(features_scaled) - self.time_steps):
            X_real.append(features_scaled[i:i+self.time_steps])

        self.X_real = np.array(X_real)

        # Predict
        self.X_pred = self.model.predict(self.X_real)

    def predict_anomalies(self, prct=98):
        if self.X_real is None or self.X_pred is None:

            warnings.warn("Data not preprocessed or predicted yet; running preprocess automatically.", UserWarning)
            self.preprocess(self.df) #, columns_to_drop=['timestamp']

            #raise ValueError("Data not preprocessed or predicted yet.")

        self.reconstruction_errors = np.mean(np.square(self.X_real - self.X_pred), axis=(1, 2))
        #threshold = np.percentile(self.reconstruction_errors, prct)
        mu = np.mean(self.reconstruction_errors)
        sigma = np.std(self.reconstruction_errors)
        threshold = mu + 3 * sigma
        anomaly_indices = np.where(self.reconstruction_errors >= threshold)[0]
        self.sorted_anomalies = anomaly_indices[np.argsort(self.reconstruction_errors[anomaly_indices])[::-1]]

        print(f"Threshold: {threshold:.6f}")
        #print(f"Top {100 - prct}% anomalies: {len(self.sorted_anomalies)} samples")
        print(f"Number of anomalies detected: {len(self.sorted_anomalies)}")
        

    def plot_anomaly(self, i):
        if self.X_real is None or self.X_pred is None:
            raise ValueError("Model not yet run on any data.")
        plt.figure(figsize=(12, 4))
        plt.plot(self.X_real[i,:,0], label='Original') # self.X_real[i,:,0].flatten(), label='Original'
        plt.plot(self.X_pred[i,:,0], label='Reconstruction')
        plt.title(f"Sample #{i} - Reconstruction Error: {self.reconstruction_errors[i]:.6f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Evaluation_{i}.png")
        plt.show()

    def top_anomalies(self, top_k: int):
        if self.sorted_anomalies is None:
            warnings.warn("No anomalies predicted yet; running predict_anomalies automatically.", UserWarning)
            self.predict_anomalies()
            #raise ValueError("Please run `predict_anomalies()` before calling `top_anomalies()`.")
        for i in self.sorted_anomalies[:top_k]:
            self.plot_anomaly(i)