# Stock Price Predictor Web Application

## Overview
This web application allows users to select a ticker symbol and a range of historical data to train a machine learning model that predicts future stock prices. Additionally, the application also plots the moving average to help visualize the stock trend.

## Features
- **Ticker Symbol Selection**: Users can choose any stock ticker symbol.
- **Date Range Selection**: Users can specify the range of historical data to be used for training.
- **Machine Learning Model**: The application trains a model based on the selected data to predict future stock prices.
- **Moving Average Plot**: Visualizes the stock trend using moving averages.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mohammad-azam22/Stock_Value_Predictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Stock_Value_Predictor
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the application:
    ```bash
    streamlit run app.py
    ```
2. Streamlit will automatically open the application in your Web browser at `http://localhost:8501`.
3. Select the desired ticker symbol and date range.
4. Click on the "Get Data" button to train the model and view the predictions.
5. The moving average plot will be displayed to visualize the stock trend.

## Technologies Used
- **Python**: For backend development.
- **Numpy**: For scientific computing.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting the moving average.
- **Scikit-learn**: For machine learning model training.
- **TensorFlow**: For model training and prediction.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
