# Telco Customer Churn Prediction

A machine learning web application that predicts customer churn through an interactive Streamlit interface, combining data preprocessing, exploratory data analysis, model evaluation, and real-time predictions.

---

## Live Demo

[Open the Telco Churn Prediction App](https://telco-churn-app-devsparkcodes.streamlit.app/)

---

## Project Overview

This project focuses on predicting customer churn using telecom customer data.

The application covers the machine learning workflow from data preprocessing and exploratory data analysis to model evaluation and interactive prediction through Streamlit.

It was built as a practical project to strengthen my understanding of machine learning, data analysis, classification, and deployment.

---

## Features

- Interactive Streamlit dashboard
- Exploratory data analysis
- Customer churn prediction
- Data preprocessing
- Feature engineering
- Model evaluation
- Interactive prediction interface
- Visual analysis of customer data

---

## How It Works

The application follows a typical machine learning workflow:

```text
Customer Data
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Classification Model
      ↓
Model Evaluation
      ↓
Customer Churn Prediction
```

The trained model is integrated into the Streamlit application to provide an interactive prediction experience.

---

## Model Performance

The deployed application achieved approximately:

- **Accuracy:** 0.81
- **F1-score for the churn class:** 0.62

These metrics are based on the model evaluation performed during the project.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and machine learning development |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Scikit-Learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Streamlit | Interactive web application and deployment |

---

## Project Structure

```text
telco-churn-streamlit-app/
│
├── assets/
│   ├── home-page.png
│   ├── eda.png
│   ├── model-performance.png
│   └── prediction-result.png
│
├── app.py
├── ohe_encoder.pkl
├── Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/devsparkcodes/telco-churn-streamlit-app.git
```

### Navigate to the Project

```bash
cd telco-churn-streamlit-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser through the local Streamlit server.

---

## Screenshots

### Home Page

![Home Page](assets/home-page.png)

### Exploratory Data Analysis

![EDA](assets/eda.png)

### Model Performance

![Model Performance](assets/model-performance.png)

### Prediction Result

![Prediction Result](assets/prediction-result.png)

---

## Learning Outcomes

Through this project, I practiced:

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Classification models
- Model evaluation
- Working with structured customer data
- Building interactive machine learning applications
- Deploying a machine learning application with Streamlit

---

## Future Improvements

- Add and compare multiple machine learning models
- Add explainable AI capabilities
- Provide detailed model comparison
- Improve prediction insights and visualizations
- Explore additional deployment options

---

## Author

**Muhammad Umar**

Building practical applications at the intersection of software engineering and AI.

- GitHub: https://github.com/devsparkcodes
- LinkedIn: https://linkedin.com/in/devsparkcodes
