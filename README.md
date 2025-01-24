# Insurance Premium and Technical Profit Forecasting

This project aims to forecast the technical profit of insurance premiums using historical data from 2013 to 2023. The project utilizes various data processing and machine learning techniques, including time series forecasting with the Prophet library.

## Project Structure
Feel free to contribute to this project by opening issues or submitting pull requests.

## Dataset
- **https://www.tsb.org.tr/tr/istatistik/genel-sigorta-verileri/saglik-kasko-analiz
- **Kasko 2019 2023.csv**: CSV file containing insurance data from 2019 to 2023.
- **Kasko_2013-2018.csv**: CSV file containing insurance data from 2013 to 2018.

### Files

- **Insurance_predict.ipynb**: Jupyter notebook containing the step-by-step process of data cleaning, visualization, and forecasting.
- **predictcode.py**: Python script that performs data cleaning, visualization, and forecasting.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- prophet

## Setup

1. Clone the repository.
2. Install the required libraries:
   ```sh
   pip install pandas numpy matplotlib statsmodels scikit-learn prophet

## Usage

### Running the Jupyter Notebook

1. Open `Insurance_predict.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the steps in the notebook to load, clean, visualize, and forecast the data.

### Running the Python Script

1. Execute the `predictcode.py` script:
   ```sh
   python predictcode.py
## Data Cleaning and Preparation

The data cleaning process involves:
- Renaming columns for clarity.
- Dropping rows with all NaN values.
- Cleaning numeric columns by removing non-numeric characters and converting them to numeric types.
- Converting the "Period" column to datetime format.
- Combining data from multiple CSV files into a single DataFrame.
- Filling missing values using forward fill.
- Removing rows with NaN or infinite values.

## Forecasting

The forecasting process involves:
- Preparing the data for Prophet by resetting the index and renaming columns.
- Initializing and fitting the Prophet model.
- Creating a future DataFrame for 5 years and making predictions.
- Plotting the forecasted technical profit with uncertainty intervals.
- Calculating and plotting the Mean Squared Error (MSE) between actual and predicted values.

## Visualization

The project includes various plots to visualize:
- Yearly Gross Written Premium, Net Written Premium, and Technical Profit.
- Forecasted Technical Profit with uncertainty intervals.
- Forecast components.
- Actual vs predicted Technical Profit with MSE.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Prophet](https://facebook.github.io/prophet/) by Facebook for time series forecasting.
- [pandas](https://pandas.pydata.org/) for data manipulation.
- [matplotlib](https://matplotlib.org/) for data visualization.
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities.

Feel free to contribute to this project by opening issues or submitting pull requests.
