
import pandas as pd
import numpy as np
import pickle


def dump_artifact(artifact, artifact_name, path):
    with open(path+artifact_name, 'wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_artifact(artifact_name, path):
    with open(path+artifact_name, 'rb') as handle:
        artifact = pickle.load(handle)
    return artifact
    