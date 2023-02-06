
import pandas as pd
import numpy as np
from conf import settings as sts
from conf import utils as uts
from conf import model_utils as muts
from conf.mymodel import MyModel
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--prices_dataframe-name", default="prices_dataframe", help="Input Prices Dataframe")
parser.add_argument("--model-name", default='model', help="Output Prediction Artifact Name")
args = vars(parser.parse_args())
#print(args)

# Import Prices Dataset
prices_dataset = uts.load_artifact(args["prices_dataframe_name"], sts.LOCAL_ARTIFACTS_PATH)

# Train-Test Split
train_dataset = prices_dataset.loc[prices_dataset['date'] < prices_dataset['date'].max()].copy()
test_dataset = prices_dataset.loc[prices_dataset['date'] >= prices_dataset['date'].max()].copy()

# Fit Model on Training Set
model = MyModel()
model.fit(train_dataset)
print("Fit model on Training Set: Done!")

# Evaluate Model on Test Set (last observed price)
error_stats = model.evaluate(test_dataset)
print("Evaluate Model on Test Set: Done!")

# Fit Model on Complete DataSet
model.fit(prices_dataset)
print(error_stats)
print("Fit model on Complete Set: Done!")

# Save Model as artifact
uts.dump_artifact(model, args["model_name"], sts.LOCAL_ARTIFACTS_PATH)