import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset
df = pd.read_csv('../data/SportPolitics.csv')

# Generate embeddings for all tweets
model = SentenceTransformer('all-mpnet-base-v2')
tweet_embeddings = model.encode(df['tweet'].values)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(tweet_embeddings, df['topic'], test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the validation set
y_pred = log_reg.predict(X_val)

# Evaluation
accuracy = accuracy_score(y_val, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary', pos_label='Sports')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
