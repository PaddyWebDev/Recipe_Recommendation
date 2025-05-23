import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

# Replace with your actual file name




data_path = os.path.join(os.path.join(os.path.dirname(__file__), 'data'), 'IndianFoodDatasetXLS.xlsx')
df = pd.read_excel(data_path)

columns = ['TranslatedRecipeName', 'Cuisine', 'TotalTimeInMins', 'TranslatedIngredients', 'Course', 'Diet']
df = df[columns].copy()

# Fill missing values
df.fillna('', inplace=True)

# Convert to string and lowercase
for col in columns:
    df[col] = df[col].astype(str).str.lower()

# Combine features into one string for vectorization
def combine_features(row):
    return ' '.join([
        row['TranslatedIngredients'],
        row['Cuisine'],
        row['Course'],
        row['Diet'],
        row['TotalTimeInMins']
    ])

df['combined_features'] = df.apply(combine_features, axis=1)

# Vectorize combined features
cv = TfidfVectorizer(stop_words='english')
vector_matrix = cv.fit_transform(df['combined_features'])
# print(cosine_similarity(vector_matrix))



joblib.dump(cv, "models/vectorizer.pkl")
joblib.dump(vector_matrix, "models/vector_matrix.pkl")
joblib.dump(df, "models/dataFrame.pkl")
