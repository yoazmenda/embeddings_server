import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
df = pd.read_csv('../data/IMDB_Dataset.csv').iloc[:6000]  # Update the file name accordingly

# Handling missing values: Remove or replace NaNs
df.dropna(inplace=True)  # Option 1: Remove rows with NaN values in 'review'
# df['review'].fillna('', inplace=True)  # Option 2: Replace NaN values with empty string

# Preprocessing function
def preprocess(text):
    # Tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text.lower())
    
    # Remove stop words
    stop_words = stopwords.words('english')
    tokens_no_stop = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_no_stop]
    
    # Stemming
    stemmer = SnowballStemmer('english')
    tokens_stemmed = [stemmer.stem(word) for word in tokens_lemmatized]
    
    return ' '.join(tokens_stemmed)

# Preprocess reviews
df['review_processed'] = df['review'].apply(preprocess)

# Vectorization
vectorizer = CountVectorizer()
X_train, X_val, y_train, y_val = train_test_split(df['review_processed'], df['sentiment'], test_size=0.2, random_state=42)
X_train_bow = vectorizer.fit_transform(X_train)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000, solver='saga')

model.fit(X_train_bow, y_train)

# Predict on the validation set
y_pred = model.predict(vectorizer.transform(X_val))

# Evaluate the model
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Accuracy Score:", accuracy_score(y_val, y_pred))
