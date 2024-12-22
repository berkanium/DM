import pandas as pd 
from sklearn.preprocessing import StandardScaler

def normalize_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    #sadece numeric kolonları normalize et
    numeric_cols=['age', 'fare']
    scaler=StandardScaler()
    df[numeric_cols]=scaler.fit_transform(df[numeric_cols])
     
    df.to_csv(output_path,index=False)
    print(f"numeric değişkenler normalize edildi, {output_path} dosyasına kaydedldi") 