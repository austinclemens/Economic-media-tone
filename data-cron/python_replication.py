# python replication of Sojourner-Bell stata code
# This file updates all the input data files and produces a new master file of data for the Soujourner-Bell analysis
import pandas as pd
import csv
from datetime import datetime
import json
import requests
from statsmodels.tsa.arima.model import ARIMA

#### SET UP THE BASE DATA ####
# set path to data folder
# path='/Users/austinc/Desktop/Media tone project/DATA/'
path='/var/www/html/media_sentiment/DATA/'

# retrieve the latest Fed tone data
p=requests.get('https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx',stream=True)
with open(path+'news_sentiment_data.xlsx','wb') as file:
	for chunk in p.iter_content(chunk_size=8192):
		file.write(chunk)

# Now load the data as a pandas dataframe
df=pd.read_excel(path+'news_sentiment_data.xlsx',sheet_name='Data',header=0)
df=df.rename(columns={'News Sentiment': 'sentiment'})

# Add year, month, and quarter variables
df['year']=df.apply(lambda row: row.date.year,axis=1)
df['quarter']=df.apply(lambda row: row.date.quarter,axis=1)

# Compute the quarterly average, create qd, a new dataframe that holds the quarterly data.
qd=df.resample('Q',on='date').mean()

# standardize the sentiment
qd['z_sentiment']=(qd['sentiment']-qd['sentiment'].mean())/qd['sentiment'].std()

#### UPDATE DJIA data using FRED ####
# load FRED API key
with open(path+'FRED_API.txt', 'r') as file:
	FRED_api=file.read()

# function to get data from the FRED API
def data_scrape(series, FRED_api, frequency):
	# get all series data in one go
	p = requests.get('https://api.stlouisfed.org/fred/series/observations?series_id='+series+'&api_key='+FRED_api+'&file_type=json&frequency='+frequency+'&aggregation_method=avg')
	json_data = json.loads(p.text)
	return json_data

# retrieve the series from FRED
DJIA_fred=data_scrape('DJIA',FRED_api,'d')['observations']

# load the existing DJIA dataset
with open(path+'DJIA_close.csv', 'r') as file:
	reader=csv.reader(file)
	DJIA=[row for row in reader][1:]

# convert all dates in both datasets to datetimes
for item in DJIA_fred:
	item['date']=datetime.strptime(item['date'], "%Y-%m-%d")

for item in DJIA:
	item[0]=datetime.strptime(item[0], "%m/%d/%y")

# collect all the dates we already have, then find those without matches from DJIA_fred
existing=[item[0] for item in DJIA]
new=[]
for item in DJIA_fred:
	if item['date'] not in existing and item['value']!='.':
		new.append([item['date'].strftime("%m/%d/%y"),item['value']])
		DJIA.append([item['date'],item['value']])

# and write the new rows to file
with open(path+'DJIA_close.csv', 'a') as file:
	writer=csv.writer(file)
	for row in new:
		writer.writerow(row)

# make the DJIA data a dataframe
dj_df=pd.DataFrame(DJIA, columns=['date', 'stock_close'])
dj_df['year']=dj_df.apply(lambda row: row.date.year,axis=1)
dj_df['quarter']=dj_df.apply(lambda row: row.date.quarter,axis=1)

# Convert the DJIA dataframe to quarterly data
dj_df['stock_close']=pd.to_numeric(dj_df['stock_close'])
qd_dj=dj_df.resample('Q',on='date').mean()

# and merge with the quarterly sentiment data
qd['date']=qd.index
merged=pd.merge(qd, qd_dj, on=['year','quarter'], how='inner')

#### GET MACRO VARIABLES ####
# retrieve data series from FRED, make dataframes, and merge with merged
macro_v=['UNRATE','GDP','CPIAUCSL','A067RO1Q156NBEA']
names={'UNRATE':'unemployment','GDP':'gdp','CPIAUCSL':'cpi','A067RO1Q156NBEA':'dpi'}
for series in macro_v:
	temp=data_scrape(series,FRED_api,'q')['observations']
	temp_df=pd.DataFrame.from_dict(temp)
	temp_df['date']=pd.to_datetime(temp_df['date'])
	temp_df['year']=temp_df.apply(lambda row: row.date.year,axis=1)
	temp_df['quarter']=temp_df.apply(lambda row: row.date.quarter,axis=1)
	temp_df=temp_df.rename(columns={'value': names[series]})
	temp_df=temp_df.drop(columns=['realtime_start', 'realtime_end','date'])
	merged=pd.merge(merged, temp_df, on=['year','quarter'], how='inner')

# convert all to numeric
columns=['sentiment','z_sentiment','stock_close','unemployment','gdp','cpi','dpi']
for var in columns:
	merged[var]=pd.to_numeric(merged[var])

# create year-over-year change for the level vars
macro_v=['gdp','cpi','stock_close']
for var in macro_v:
	merged['ppchg_'+var]=100*merged[var].pct_change(periods=4)

# some final data cleaning and segment the data to train on just 1988-2016 data
merged['date_q'] = pd.PeriodIndex(year=merged['year'], quarter=merged['quarter'], freq='Q').to_timestamp()
merged.index=merged['date_q']
merged.index.freq='QS-OCT'
merged=merged.dropna()
merged_training=merged[(merged['year']>1987) & (merged['year']<2017)].copy()

# run the ARIMA model - the code below perfectly replicates results of the Sojourner-Bell model
model = ARIMA(endog=merged_training['z_sentiment'], exog=merged_training[['ppchg_gdp','unemployment','ppchg_cpi','ppchg_stock_close']],order=(1,0,4))
model_fit = model.fit(method='innovations_mle')
print(model_fit.summary())

# forecast on the original merged dataset, predict on the training dataset, then concat those two dataframes
merged_testing=merged[merged['year']>=2017].copy()
# forecast_values = model_fit.forecast(steps=32, exog=merged_testing[['ppchg_gdp','unemployment','ppchg_cpi','ppchg_stock_close']],dynamic=False) 
# forecast_values = model_fit.predict(steps=32, start='2017-01-01',end='2024-10-01',endog=merged_training['z_sentiment'],exog=merged_testing[['ppchg_gdp','unemployment','ppchg_cpi','ppchg_stock_close']],dynamic=False) 
fore=model_fit.apply(endog=merged_testing['z_sentiment'],exog=merged_testing[['ppchg_gdp','unemployment','ppchg_cpi','ppchg_stock_close']])
merged_testing['yhat']=fore.fittedvalues
merged_testing['residuals']=merged_testing['z_sentiment']-merged_testing['yhat']

predict_values = model_fit.predict(exog=merged_training[['ppchg_gdp','unemployment','ppchg_cpi','ppchg_stock_close']],dynamic=False) 
merged_training['yhat']=predict_values
merged_training['residuals']=merged_training['z_sentiment']-merged_training['yhat']

# save the forecast dataset
final_df=pd.concat([merged_training,merged_testing])
final_df=final_df.drop(columns=['sentiment', 'date','stock_close','gdp','cpi','date_q'])
final_df.to_csv(path+'forecast_data.csv', index=False)

