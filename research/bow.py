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
df = pd.read_csv('../data/SportPolitics.csv')

# Display the first few rows of the dataframe
print(df.head())

# Plot the distribution of topics
sns.countplot(x='topic', data=df)
plt.title('Distribution of Topics in original dataset')
plt.show()


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

# preprocess tweets
df['tweet_processed'] = df['tweet'].apply(preprocess)

# Visualize the preprocessing steps for a single tweet example
example_tweet = df['tweet'].iloc[60]
print("Original Tweet:\n", example_tweet)
print("\nProcessed Tweet:\n", preprocess(example_tweet))

# Vectorization
vectorizer = CountVectorizer()
X_train, X_val, y_train, y_val = train_test_split(df['tweet_processed'], df['topic'], test_size=0.2, random_state=42)
X_train_bow = vectorizer.fit_transform(X_train)

# Display the feature names and the shape of the resulting matrix
print("Feature names (sample):", vectorizer.get_feature_names_out()[0:10])
print("Shape of Bag of Words matrix:", X_train_bow.shape)

# Display the Bag of Words matrix for the first few tweets
pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out()).head()


# Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_bow, y_train)

# Predict on the validation set
y_pred = model.predict(vectorizer.transform(X_val))

# Evaluate the model
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Accuracy Score:", accuracy_score(y_val, y_pred))

