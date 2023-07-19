import os
import pandas as pd
from conf import settings as sts
from conf import utils as uts
import boto3
from datetime import datetime
from dateutil.relativedelta import relativedelta
import nltk

# Download NLTK tokenizer data for tokenization (used in clean_text function)
nltk.download('punkt')

# Load ENV secrets (AWS credentials and bucket name)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Create an S3 client using AWS credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Consolidate Dataset from Remote Separate Excel Files
_dataframe_list = []

# Loop for every file URL path
for i, url in enumerate(sts.REMOTE_FILE_PATHS):
    # Set the remote path for the first URL, and use the specified URL path for the rest
    if i == 0:
        remote_path = "https://www.dane.gov.co/files/operaciones/SIPSA/"
    else:
        remote_path = sts.REMOTE_URL_PATH

    # Load Excel file and preprocess it using the load_dataset function
    dataframe = uts.load_dataset(url, local_path=sts.LOCAL_DATA_PATH, url_path=remote_path)
    _dataframe_list.append(dataframe)

# Concatenate all pandas DataFrames and rename columns
prices_dataframe = pd.concat(_dataframe_list, ignore_index=True)
prices_dataframe.columns = ['date', 'group', 'product', 'market', 'mean_price']

# Sort the DataFrame by 'group', 'product', 'market', and 'date'
prices_dataframe.sort_values(['group', 'product', 'market', 'date'], inplace=True, ignore_index=True)

print("Consolidate historical data: Done!")

# Convert the 'date' column to datetime format
prices_dataframe['date'] = pd.to_datetime(prices_dataframe['date'])

# Get the cutoff date (last date + 1 month)
cutoff_date = prices_dataframe['date'].max() + relativedelta(months=1)

# Load the dataset for the last month using the load_last_month_dataset function
last_month_prices_dataframe = uts.load_last_month_dataset(cutoff_date)

print("Get data after historical cutoff: Done!")

# Concatenate the prices_dataframe with last_month_prices_dataframe, keeping only common columns
prices_dataframe = pd.concat([prices_dataframe, last_month_prices_dataframe], join='inner', ignore_index=True)
prices_dataframe.drop_duplicates(subset = ['date', 'product', 'market'], inplace=True)

# Preprocess text in 'product' and 'market' columns by cleaning the text
prices_dataframe['product'] = prices_dataframe['product'].apply(uts.clean_text)
prices_dataframe['market'] = prices_dataframe['market'].apply(uts.clean_text)

print("Data processing: Done!")

# Set monthly date range for all time series by reindexing the pivoted dataframe
date_range = pd.date_range(start=prices_dataframe['date'].min(), end=prices_dataframe['date'].max(), freq='MS')
prices_dataframe_piv = prices_dataframe.pivot(index='date', columns=['product', 'market'], values='mean_price')
prices_dataframe_piv = prices_dataframe_piv.reindex(date_range)

# Select Columns (Time series) with a minimum number of observations in the last 3 years
selected_series = prices_dataframe_piv.dropna(axis=1).columns
print(f"Selected time series: {len(selected_series)} out of {prices_dataframe_piv.shape[1]} in total")
prices_dataframe_piv = prices_dataframe_piv.loc[:, selected_series]

# Unstack MultiIndex and rename columns
prices_dataframe = prices_dataframe_piv.unstack().reset_index(['product', 'market']).reset_index()
prices_dataframe.columns = ['date', 'product', 'market', 'mean_price']

print("Batch reindexing and series selection: Done!")

# Save DataFrame to S3 Bucket
date_id = datetime.now().strftime(format="%Y%m%d%H%M%S")

LOCAL_FILE_PATH = sts.LOCAL_DATA_PATH + f'prices_dataframe_{date_id}.parquet'
S3_FILE_PATH = sts.S3_PROJECT_PATH + f'prices_dataframe_{date_id}.parquet'

# Save the DataFrame to a Parquet file
prices_dataframe.to_parquet(LOCAL_FILE_PATH, index=False)

# Upload the Parquet file to the S3 bucket
s3_client.upload_file(LOCAL_FILE_PATH, AWS_BUCKET_NAME, S3_FILE_PATH)

# Remove the local file after saving it to S3
os.remove(LOCAL_FILE_PATH)

print("Dataframe saved in S3: Done!")
