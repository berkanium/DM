import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models(input_path, output_path):
    df = pd.read_csv(input_path)
    
    #özellikler ve hedef değişkenler
    X=df.drop('survived',axis=1)
    y=df['survived']
     
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
    results={}
    
     #KNN 3,7,11
     
    knn_neigbors=[3,7,11]
    for n__neighbors in knn_neigbors:
        knn=KNeighborsClassifier(n_neighbors=n__neighbors)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        results[f'KNN-{n__neighbors}']={
            'accuracy': accuracy_score(y_test,y_pred),
            'precision': precision_score(y_test,y_pred),
            'recall': recall_score(y_test,y_pred),
            'f1': f1_score(y_test,y_pred) 
        }
        
    #MLP 1,2 ve 3 hidden layrs 
    hidden_layer_configs =[(32,),(32,32),(32,32,32)]
    for layers in hidden_layer_configs:
        mlp=MLPClassifier(hidden_layer_sizes=layers,max_iter=500, random_state=46)
        mlp.fit(X_train,y_train)
        y_pred=mlp.predict(X_test)
        results[f'MLP-{layers}-layers']={
            'accuracy': accuracy_score(y_test,y_pred),
            'precision': precision_score(y_test,y_pred),
            'recall': recall_score(y_test,y_pred),
            'f1': f1_score(y_test,y_pred) 
        }
        
    #Naive Bayes (NB)
    nb=GaussianNB()
    nb.fit(X_train,y_train)
    y_pred=nb.predict(X_test)
    results[f'Naive Bayes']={
            'accuracy': accuracy_score(y_test,y_pred),
            'precision': precision_score(y_test,y_pred),
            'recall': recall_score(y_test,y_pred),
            'f1': f1_score(y_test,y_pred) 
        }
    
    with open(output_path,'w') as f:
        for name, metrics in results.items():
            f.write(f"{name}: {metrics}\n") 
     
    print(f"model sonuçları {output_path} dosyasına yazdırıldı.")