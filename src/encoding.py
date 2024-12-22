import pandas as pd 
from sklearn.preprocessing import LabelEncoder

def encode_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    #Label encoding işlemleri
    encoder = LabelEncoder()
    df['sex']= encoder.fit_transform(df['sex'])
    df['embarked']= encoder.fit_transform(df['embarked'])
     
    df.to_csv(output_path, index=False)    
    print(f"Kategorik değişkenler encode edildi, {output_path} kaydedildi.")