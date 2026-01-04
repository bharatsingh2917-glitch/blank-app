# ❤️ Heart Disease Prediction App

A machine learning-powered Streamlit web application for predicting heart disease based on patient medical parameters.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Features

- **Interactive Web Interface**: User-friendly Streamlit app for easy input of patient data
- **Machine Learning Model**: Logistic Regression model trained on heart disease dataset
- **Real-time Predictions**: Instant predictions based on 13 medical parameters
- **Educational Tool**: Great for learning about heart disease risk factors

## Medical Parameters Used

The model considers the following 13 parameters:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression Induced by Exercise
- Slope of Peak Exercise ST Segment
- Number of Major Vessels Colored by Fluoroscopy
- Thalassemia

## Installation & Usage

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bharatsingh2917-glitch/blank-app.git
   cd blank-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## Deployment

### Streamlit Cloud (Recommended)

1. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and sign in with your GitHub account.
2. **Click "New app"**.
3. **Select your repository**: `bharatsingh2917-glitch/blank-app`.
4. **Choose branch**: `main`.
5. **Main file path**: `streamlit_app.py`.
6. **Click "Deploy!"**.

Your app will be live at a URL like `https://your-app-name.streamlit.app/`.

### Other Platforms

- **Heroku**: Create a `Procfile` with `web: streamlit run streamlit_app.py --server.port $PORT --server.headless true`.
- **AWS/Docker**: Containerize the app and deploy to your preferred cloud provider.

## Model Performance

- **Algorithm**: Logistic Regression
- **Dataset**: UCI Heart Disease Dataset (sample data included)
- **Accuracy**: ~85% on test data

## Disclaimer

⚠️ **Important**: This application is for educational and demonstration purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## Project Structure

```
├── streamlit_app.py      # Main Streamlit application
├── data.csv             # Heart disease dataset
├── requirements.txt     # Python dependencies
├── Project_10_Heart_Disease_Prediction.ipynb  # Original Jupyter notebook
└── README.md           # This file
```

## Technologies Used

- **Streamlit**: Web app framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## Contributing

Feel free to fork this repository and submit pull requests with improvements!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
