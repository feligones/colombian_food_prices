import os
from dotenv import load_dotenv

import pandas as pd
from conf import settings as sts
from conf import utils as uts
import boto3
from datetime import datetime
import nltk
nltk.download('punkt')

# Load ENV secrets
assert load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Consolidate Dataset from Remote Separate Excel Files
_dataframe_list = []

# Loop for every file URL path
for url in sts.REMOTE_FILE_PATHS:
    # Load Excel file and preprocess it
    dataframe = uts.load_dataset(url, local_path=sts.LOCAL_DATA_PATH, url_path=sts.REMOTE_URL_PATH)
    _dataframe_list.append(dataframe)

# Concat all pandas DFs and name columns
prices_dataframe = pd.concat(_dataframe_list, ignore_index=True)
prices_dataframe.columns = ['date', 'group', 'product', 'market', 'mean_price']

prices_dataframe.sort_values(['group', 'product', 'market', 'date'], inplace=True, ignore_index=True)

print("Consolidate dataframes: Done!")

# Set format for Date
prices_dataframe['date'] = pd.to_datetime(prices_dataframe['date'])

# Extract market department/municipality
# prices_dataframe['department'] = prices_dataframe['market'].str.split(',',expand=True)[0]

# Preprocess text from string columns: group, product, market and department
prices_dataframe['group'] = prices_dataframe['group'].apply(uts.clean_text)
prices_dataframe['product'] = prices_dataframe['product'].apply(uts.clean_text)
prices_dataframe['market'] = prices_dataframe['market'].apply(uts.clean_text)

print("Data processing: Done!")

# Set monthly date range for all the time series by reindexing the pivoted dataframe
date_range = pd.date_range(start = prices_dataframe['date'].min(), end = prices_dataframe['date'].max(), freq = 'MS')
prices_dataframe_piv = prices_dataframe.pivot(index='date', columns = ['group', 'product', 'market'], values = 'mean_price')
prices_dataframe_piv = prices_dataframe_piv.reindex(date_range)

# Select Columns (Timeseries) with minimum observations 
# (Condition: all monthly datapoints in the last 3 years)
selected_series = prices_dataframe_piv.dropna(axis = 1).columns
print(f"Selected time series : {len(selected_series)} out of {prices_dataframe_piv.shape[1]} in total")
prices_dataframe_piv = prices_dataframe_piv.loc[:, selected_series]

# Unstack Multindex and rename columns
prices_dataframe = prices_dataframe_piv.unstack().reset_index(['group', 'product', 'market']).reset_index()
prices_dataframe.columns = ['date', 'group', 'product', 'market', 'mean_price']

print("Batch reindexing and series selection: Done!")

# Save DataFrame to S3 Bucket
date_id = datetime.now().strftime(format = "%Y%m%d%H%M%S")

LOCAL_FILE_PATH = sts.LOCAL_DATA_PATH + f'prices_dataframe_{date_id}.csv'
S3_FILE_PATH = sts.S3_PROJECT_PATH + f'prices_dataframe_{date_id}.csv'

prices_dataframe.to_csv(LOCAL_FILE_PATH, index=False)

s3_client.upload_file(LOCAL_FILE_PATH, AWS_BUCKET_NAME, S3_FILE_PATH)

print("Dataframe saved in S3: Done!")