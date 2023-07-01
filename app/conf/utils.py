
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
    """
    This method saves an object as a pickle in a specified local path.

    Args:
        artifact (object): oject t be saved as artifact.
        artifact_name (string): name of the saved artifact.
        path (string): local path to save artifacts.
    """

    with open(path+artifact_name+'.pkl', 'wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_artifact(artifact_name, path):
    """
    This method returns the queried artfact in its original form.

    Args:
        artifact_name (string): name of the saved artifact.
        path (string): local path to collect artifacts.
    
    Returns:
        artifact (object): object saves as the queried artifact.
    """

    with open(path+artifact_name+'.pkl', 'rb') as handle:
        artifact = pickle.load(handle)
    return artifact

def load_dataset(filename, local_path, url_path):
    """
    This method returns consolidated prices dataframe.

    The process consists of requesting an XLSX file in the DANE webpage 
    corresponding to one or multiple years of monthly average prices.

    Args:
        filename (string): name of the XLSX file.
        local_path (string): local path to save the requested file.
        url_path (string): remote path to request the XLSX file.

    Returns:
        concat_dataframe (pd.Dataframe): pandas dataframe form of the 
                                         requested XLSX file.
    """

    # Create path to save requested XLSX file
    file_path = local_path + filename
    
    # Check if file is not already saved locally
    if filename not in os.listdir(local_path):
        # Create complete request URL
        url = url_path + filename
        # Request file on URL
        resp = requests.get(url)
        assert resp.status_code == 200, resp.status_code + " - " + filename
        # Save requested file locally
        file = open(file_path, 'wb')
        file.write(resp.content)
        file.close()
    # Instance a workbook
    workbook = load_workbook(file_path, data_only = True)
    # Get sheetnames on workbook
    sheetnames = workbook.sheetnames

    _dataframe_list = []
    # For each sheetname...
    for sheet in sheetnames[1:]:
        # Create pd Dataframe with values
        dataframe = pd.DataFrame(workbook[sheet].values)
        # Clean useless rows and columns
        _valid_index = pd.to_datetime(dataframe[0], errors='coerce').dropna(axis = 0).index
        dataframe = dataframe.filter(_valid_index, axis = 0).dropna(axis = 1)
        _dataframe_list.append(dataframe)
    
    # Concatenate all dataframes of each sheet
    concat_dataframe = pd.concat(_dataframe_list, ignore_index=True)
    
    return concat_dataframe

def clean_text(text):
    """
    This methods returns the input text without accents, 
    special characters and lower cased.

    Args:
        text (string): input text.

    Returns:
        clean_text (string): processed text.
    """

    clean_text = unidecode(text.lower())
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = ' '.join(word_tokenize(clean_text))
    return clean_text