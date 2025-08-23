# Book Rating Prediction Project

This project aims to predict the average rating of a book based on its metadata. The repository contains a Jupyter Notebook detailing the data exploration, feature engineering, and model training process, as well as a web application built with Streamlit to serve the final model for real-time predictions.

## 🚀 Features

* **In-depth Data Analysis:** The `Project ML.ipynb` notebook covers all steps from data cleaning and exploratory data analysis to feature engineering and model training.

* **Interactive Web Application:** A user-friendly web app built with Streamlit that allows for:

  * **Single Predictions:** Enter a book's details manually to get an instant rating prediction.

  * **Batch Predictions:** Upload a CSV file with multiple books to get predictions for all rows.

## 🛠️ Technology Stack

* **Backend:** Python

* **Data Manipulation & Analysis:** pandas, NumPy

* **Machine Learning:** scikit-learn, LightGBM, XGBoost etc.

* **Web Framework:** Streamlit

* **Model & Object Serialization:** joblib

## 📂 Project Structure

```
.
├── Data/ # Folder containing data used for training 
├── Models/                 # Folder containing serialized models and objects (.joblib files)
├── WebApp/                 # Contains the Streamlit application
├── .gitignore              # Specifies files and folders to be ignored by Git
├── Project ML.ipynb        # Jupyter Notebook with the full ML workflow
├── requirements.txt        # Lists the project's Python dependencies
└── README.md               # This file
```

## 🏃‍♀️ How to Run the Web Application

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KhalilSRIDI/S25-ML-Project-Book-Rating-Prediction-DSTI/
   cd S25-ML-Project-Book-Rating-Prediction-DSTI
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   
   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
4. 
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app:**

   ```bash
   streamlit run WebApp/webapp.py
   ```

6. Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
