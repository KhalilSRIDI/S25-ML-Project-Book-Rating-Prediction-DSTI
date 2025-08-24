import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from scipy.sparse import hstack
from datetime import datetime
import io
import csv

st.set_page_config(
    page_title="Book Rating Prediction",
    layout="wide"
)

MODELS_FOLDER = "./Models/"

@st.cache_resource
def load_objects():
    try:
        model = joblib.load(f'{MODELS_FOLDER}best_model.joblib')
        feature_columns = joblib.load(f'{MODELS_FOLDER}column_list.joblib')
        scaler = joblib.load(f'{MODELS_FOLDER}scaler.joblib')
        author_rating_map = joblib.load(f'{MODELS_FOLDER}author_avg_rating.joblib')
        publisher_rating_map = joblib.load(f'{MODELS_FOLDER}publisher_avg_rating.joblib')
        publisher_map = joblib.load(f'{MODELS_FOLDER}publisher_map.joblib')
        return model, feature_columns,scaler, author_rating_map, publisher_rating_map, publisher_map
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}. Please ensure the '.joblib' files are in the '{MODELS_FOLDER}' directory.")
        return None, None, None, None, None
    
model, feature_columns, scaler, author_rating_map, pub_rating_map, pub_map = load_objects()

normalized_publishers = sorted(list(set(pub_map.values())))
normalized_publishers.append("Other")

def clean_author(author):
    author = re.sub(r'[^\w\s]', '', author)
    author = author.lower()
    author = re.sub(r'\s+', ' ', author)
    return author.strip()

def longest_common_prefix(strs):
    if not strs:
        return ""
    
    shortest_str = min(strs, key=len)
    for i, char in enumerate(shortest_str):
        for other in strs:
            if i >= len(other) or other[i] != char:
                return shortest_str[:i]
    return shortest_str

def create_series(df):
    collection_pattern = re.compile(
        r'Boxed Set|Box Set|Collection|Complete Series|Books\s\d+-\d+|#\d+-\d+', 
        re.IGNORECASE
    )
    df['is_collection'] = df['title'].str.contains(collection_pattern, na=False).astype(int)
    df['author_primary'] = df['authors'].str.split('/').str[0].str.strip()
    number_pattern = re.compile(r'\(?#\s*(\d+)\)?|Book\s+(\d+)|Volume\s+(\d+)|Part\s+(\d+)', re.IGNORECASE)
    book_numbers = df['title'].str.extract(number_pattern)
    is_series_regex = book_numbers.notna().any(axis=1)
    author_titles = df.groupby('author_primary')['title'].apply(list)
    authors_with_multiple_books = author_titles[author_titles.apply(len) > 1].index
    series_prefixes = {}
    for author in authors_with_multiple_books:
        titles = author_titles[author]
        prefix = longest_common_prefix(titles)
        if len(prefix) > 5:
            series_prefixes[author] = prefix.strip()
    df['series_name'] = df['author_primary'].map(series_prefixes)
    is_series_prefix = df['series_name'].notna()
    provisional_is_series = (is_series_regex | is_series_prefix).astype(int)
    df['is_series'] = np.where(df['is_collection'] == 1, 0, provisional_is_series)
    df.drop(columns=['author_primary', 'series_name'], inplace=True)
    df.drop(columns=['series_book_count', 'book_number_in_series'], errors='ignore', inplace=True)
    return df

def process_and_predict(df_input, model, feature_columns, scaler, author_rating_map, pub_rating_map, pub_map):
    df_input = create_series(df_input)
    df_input['publication_date'] = pd.to_datetime(df_input['publication_date'])
    df_input['book_age'] = 2025 - df_input['publication_date'].dt.year
    df_input['main_author'] = df_input['authors'].str.split('/').str[0]
    df_input['main_author'] = df_input['main_author'].apply(clean_author)
    df_input['author_avg_rating'] = df_input['main_author'].map(author_rating_map).fillna(0)
    df_input["publisher_normalized"] = df_input["publisher"].replace(pub_map)
    df_input["publisher_avg_rating"] = df_input["publisher_normalized"].map(pub_rating_map).fillna(0)

    df_predict = pd.DataFrame(index=df_input.index)
    df_predict['is_collection'] = df_input['is_collection']
    df_predict['is_series'] = df_input['is_series']
    df_predict['author_avg_rating'] = df_input['author_avg_rating']
    df_predict["credibility"] = np.log10(df_input["ratings_count"] + 1) + np.log10(df_input["text_reviews_count"] + 1)
    df_predict['publisher_avg_rating'] = df_input['publisher_avg_rating']
    df_predict['num_pages'] = df_input['num_pages']
    df_predict['book_age'] = df_input['book_age']
    numerical_features_to_scale = [
    'num_pages',
    'credibility',
    'author_avg_rating', 
    'publisher_avg_rating', 
    'book_age'
    ] 
    df_predict[numerical_features_to_scale] = scaler.transform(df_predict[numerical_features_to_scale])
    df_predict = df_predict[feature_columns]
    
    
    predictions = model.predict(df_predict)
    clipped_predictions = np.clip(predictions, 1, 5)
    
    return pd.Series(clipped_predictions, index=df_input.index)


st.title('Book Rating Prediction')

if model is None:
    st.warning("Application is not ready. Please check file loading errors.")
else:
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    with tab1:
        st.header("Enter Book Details")
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Book Title", "The Hobbit")
                authors = st.text_input("Author(s)", "Stephen King")
                selected_publisher = st.selectbox(
                    "Publisher",
                    options=normalized_publishers,
                    index=normalized_publishers.index("Penguin")
                )
                publication_date_str = st.text_input("Publication Date (DD/MM/YYYY)", "10/01/2005")
            
            with col2:
                num_pages = st.number_input('Number of Pages', min_value=0, max_value=6576, value=260)
                ratings_count = st.number_input('Number of Ratings', min_value=0, max_value=4597666, value=2115)
                text_reviews_count = st.number_input('Number of Text Reviews', min_value=0, max_value= 94265, value=114)

            submit_button = st.form_submit_button('Predict Rating', use_container_width=True)

        if submit_button:
            try:
                datetime.strptime(publication_date_str, '%d/%m/%Y')
                with st.spinner('Predicting'):
                    raw_data = {
                        'title': [title], 'authors': [authors], 'publisher': [selected_publisher],
                        'publication_date': [publication_date_str], 'num_pages': [num_pages],
                        'ratings_count': [ratings_count], 'text_reviews_count': [text_reviews_count]
                    }
                    df_input = pd.DataFrame(raw_data)
                    
                    prediction = process_and_predict(df_input, model, feature_columns, scaler, author_rating_map, pub_rating_map, pub_map)
                    
                    st.subheader('Prediction Result')
                    st.success(f"The predicted book rating is **{prediction.iloc[0]:.2f} / 5**")
                    
            except ValueError:
                st.error("Check input")

    with tab2:
        st.header("Upload a CSV File for Batch Predictions")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                sample = uploaded_file.read(2048).decode("utf-8", errors="ignore")
                uploaded_file.seek(0)
                try:
                    delimiter = csv.Sniffer().sniff(sample).delimiter
                except Exception:
                    delimiter = ","   
                    
                df_batch = pd.read_csv(uploaded_file, sep=delimiter, engine="python",quotechar='"')
                if st.button('Run Batch Prediction', use_container_width=True):
                    with st.spinner('Predicting ratings...'):
                        # Keep a copy of the original relevant columns
                        prediction__df = df_batch[['title', 'authors', 'publisher', 'publication_date', 'num_pages', 'ratings_count', 'text_reviews_count']].copy()
                        predictions = process_and_predict(df_batch.copy(), model, feature_columns, scaler, author_rating_map, pub_rating_map, pub_map)
                        # Add predictions to the original input dataframe
                        prediction__df['predicted_rating'] = predictions.round(2)
                        
                        st.subheader("Prediction Results")
                        st.dataframe(prediction__df)
                        
                        csv = prediction__df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                           label="Download Results as CSV",
                           data=csv,
                           file_name='book_ratings_predictions.csv',
                           mime='text/csv',
                           use_container_width=True
                        )
            except Exception as e:
                st.error(f"An error occurred: {e}")