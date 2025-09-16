from os import path ,listdir
import pandas as pd
path_to_modules = path.join("code")
import sys
sys.path.append(path_to_modules)

from Evaluation import Evaluation
from ReadCloudVM import getDataVm 
print("Is running...")





#CloudPath =r"Arctur_Data"
cloud_model_path = r"Models\Cloud_Model_folder\LSTM_autoencoder_ICCS_final"
#path2gb = r"Data\data2gb_LSTM.csv"
#path4gb = r"Data\data4gb_LSTM.csv"

#GetData = getDataVm(CloudPath)


#df_original = GetData.get_first_device("vm101")

#print("Next Data",GetData.next_vm)

#print("shape of df:",df_original.shape)

#df_github = df_original[:1000]

df_github = pd.read_csv(r"data_github.csv")

#df_github.to_csv("data_github.csv",index=False)
# Load the Model
print(f"columns: {df_github.columns}")
df_model = df_github.drop(columns=['device_id','timestamp']).copy()
anomaly_Detector = Evaluation(df_model)
anomaly_Detector.load_a_model(cloud_model_path)


anomaly_Detector.predict_anomalies(prct=98)

# Access the scores:
errors = anomaly_Detector.reconstruction_errors
anomalies = anomaly_Detector.sorted_anomalies

print("First 5 reconstruction errors:", errors[:5])
print("Top anomalies indices:", anomalies[:10])
anomaly_Detector.plot_anomaly(1)