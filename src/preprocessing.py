import pandas as pd 

def preprocess_data(input_path, output_path):
    df= pd.read_csv(input_path)
    
    #eksik değerleri
    df['age']=df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna('S')
    
    #işlenmiş veri kaydet 
    df.to_csv(output_path,index=False)
    print(f"Veri ön işleme tamamlandı, {output_path} dosyasına kaydedildi.")