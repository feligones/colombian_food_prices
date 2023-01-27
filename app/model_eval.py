
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
from conf import settings as sts
from conf import utils as uts
from conf import model_utils as muts

import sys
print(sys.path)

train_model = uts.load_artifact('train_model', sts.LOCAL_ARTIFACTS_PATH)