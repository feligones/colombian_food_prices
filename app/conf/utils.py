
import pandas as pd
import numpy as np
import pickle
import requests
from openpyxl import load_workbook
import os
from unidecode import unidecode
from nltk.tokenize import word_tokenize
import re

def dump_artifact(artifact, artifact_name, path):
    with open(path+artifact_name, 'wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_artifact(artifact_name, path):
    with open(path+artifact_name, 'rb') as handle:
        artifact = pickle.load(handle)
    return artifact

def load_dataset(filename, local_path, url_path):
    
    file_path = local_path + filename
    
    if filename not in os.listdir(local_path):
    
        url = url_path + filename
        # print(url)
        resp = requests.get(url)
        assert resp.status_code == 200, resp.status_code

        file = open(file_path, 'wb')
        file.write(resp.content)
        file.close()
    
    workbook = load_workbook(file_path, data_only = True)
    sheetnames = workbook.sheetnames
    
    _dataframe_list = []
    
    for sheet in sheetnames[1:]:
        dataframe = pd.DataFrame(workbook[sheet].values)
        _valid_index = pd.to_datetime(dataframe[0], errors='coerce').dropna(axis = 0).index
        dataframe = dataframe.filter(_valid_index, axis = 0).dropna(axis = 1)
        _dataframe_list.append(dataframe)
    
    concat_dataframe = pd.concat(_dataframe_list, ignore_index=True)
    
    return concat_dataframe

def clean_text(text):
    clean_text = unidecode(text.lower())
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = ' '.join(word_tokenize(clean_text))
    return clean_text