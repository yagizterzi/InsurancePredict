import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from prophet import Prophet

files = [r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\insurance\Kasko 2019 2023.csv',r'C:\Users\yagiz\OneDrive\Masaüstü\Uygulamalar\kodlar\insurance\Kasko_2013-2018.csv']

# Function to clean and prepare each file
def clean_file(file_path):
    # Load the file
    df = pd.read_csv(file_path)

    # Rename columns for clarity
    df.columns = [
        "Period",
        "Gross_Written_Premium_TL",
        "Gross_Increase_Rate",
        "Net_Written_Premium_TL",
        "Net_Increase_Rate",
        "Net_Earned_Premium_TL",
        "Earned_Increase_Rate",
        "Net_Realized_Loss_TL",
        "Loss_Increase_Rate",
        "Technical_Profit_TL",
        "Technical_Profit_Increase_Rate",
        "Net_Loss_Premium_Rate",
        "Combined_Ratio"
    ]

    # Drop rows with all NaN values
    df = df.dropna(how='all').reset_index(drop=True)

    # Clean numeric columns
    numeric_columns = [
        "Gross_Written_Premium_TL",
        "Gross_Increase_Rate",
        "Net_Written_Premium_TL",
        "Net_Increase_Rate",
        "Net_Earned_Premium_TL",
        "Earned_Increase_Rate",
        "Net_Realized_Loss_TL",
        "Loss_Increase_Rate",
        "Technical_Profit_TL",
        "Technical_Profit_Increase_Rate",
        "Net_Loss_Premium_Rate",
        "Combined_Ratio"
    ]

    for col in numeric_columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d.,()-]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.replace(r"\((.*?)\)", r"-\1", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert "Period" to datetime
    df["Period"] = pd.to_datetime(df["Period"], format='%Y', errors='coerce')

    return df

# Process and combine all files
dataframes = [clean_file(file) for file in files]
combined_data = pd.concat(dataframes, ignore_index=True)

# Set "Period" as the index and sort
combined_data = combined_data.set_index("Period").sort_index()

# Fill missing values (forward fill)
combined_data = combined_data.ffill()

# Ensure no NaN or infinite values
combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()

# Remove rows where the index is NaT
combined_data = combined_data[combined_data.index.notna()]



plt.figure(figsize=(14, 7))
plt.plot(combined_data.index, combined_data['Gross_Written_Premium_TL'] / 10**9, label='Gross Written Premium TL (in billions)')
plt.plot(combined_data.index, combined_data['Net_Written_Premium_TL'] / 10**9, label='Net Written Premium TL (in billions)')
plt.plot(combined_data.index, combined_data['Technical_Profit_TL'] / 10**9, label='Technical Profit TL (in billions)')
plt.xlabel('Year')
plt.ylabel('Amount (in billions TL)')
plt.title('Yearly Gross Written Premium, Net Written Premium, and Technical Profit')
plt.legend()
plt.show()

# Ensure no NaN or infinite values before decomposition
combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()


# Remove any remaining NaN or infinite values
combined_data = combined_data.dropna()


# Ensure the index is a datetime index and sorted
if not isinstance(combined_data.index, pd.DatetimeIndex):
    combined_data.index = pd.to_datetime(combined_data.index)
combined_data = combined_data.sort_index()

# Ensure the index has a frequency
combined_data.index = pd.to_datetime(combined_data.index)
combined_data.index.freq = pd.infer_freq(combined_data.index)

# Prepare data for Prophet
prophet_data = combined_data.reset_index()[['Period', 'Technical_Profit_TL']]
prophet_data.columns = ['ds', 'y']

# Initialize and fit the Prophet model
model = Prophet()
model.fit(prophet_data)

# Make future dataframe for 5 years
future = model.make_future_dataframe(periods=5*12, freq='MS')
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Technical Profit Forecast with Uncertainty Intervals')
plt.xlabel('Year')
plt.ylabel('Technical Profit TL (in billions)')
plt.show()

# Plot the forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Plot the forecasted data in billions
plt.figure(figsize=(14, 7))
plt.plot(forecast['ds'], forecast['yhat'] / 10**9, label='Forecasted Technical Profit TL (in billions)')
plt.fill_between(forecast['ds'], forecast['yhat_lower'] / 10**9, forecast['yhat_upper'] / 10**9, color='gray', alpha=0.2, label='Uncertainty Interval')
plt.xlabel('Year')
plt.ylabel('Technical Profit TL (in billions)')
plt.title('Forecasted Technical Profit with Uncertainty Intervals')
plt.legend()
plt.show()

# Calculate and plot MSE
# Ensure the actual data and forecast data are aligned
actual = prophet_data.set_index('ds')['y']
predicted = forecast.set_index('ds')['yhat']
common_dates = actual.index.intersection(predicted.index)
actual_common = actual.loc[common_dates]
predicted_common = predicted.loc[common_dates]

# Ensure the lengths are consistent
if len(actual_common) > len(predicted_common):
    actual_common = actual_common.iloc[:len(predicted_common)]
elif len(predicted_common) > len(actual_common):
    predicted_common = predicted_common.iloc[:len(actual_common)]

mse = mean_squared_error(actual_common, predicted_common)

plt.figure(figsize=(14, 7))
plt.plot(common_dates, actual_common / 10**9, label='Actual Technical Profit TL (in billions)')
plt.plot(common_dates, predicted_common / 10**9, label='Predicted Technical Profit TL (in billions)')
plt.xlabel('Year')
plt.ylabel('Technical Profit TL (in billions)')
plt.title(f'Mean Squared Error: {mse:.2f}')
plt.legend()
plt.show()