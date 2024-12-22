from src.preprocessing import preprocess_data
from src.encoding import encode_data
from src.normalization import normalize_data
from src.models import train_models


raw_data="data/veri_kumesi.csv"
preprocessed_data="data/veri_kumesi.csv"
encoded_data="data/veri_kumesi.csv"
normalized_data="data/veri_kumesi.csv"
results_file="output/model_results.txt"

#işleme kodları burada yer alır

preprocess_data(raw_data,preprocessed_data)
encode_data(preprocessed_data,encoded_data) 
normalize_data(encoded_data, normalized_data)
train_models(normalized_data,results_file)
 