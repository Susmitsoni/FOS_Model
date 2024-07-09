import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# # Enable inline plotting for Jupyter notebook
# %matplotlib inline

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('C:/programming/python/DataSet/New folder/FOS_CSV.csv', encoding='latin1')

# Preprocessing: Tokenization and Vectorization
stop_words = set(stopwords.words('english'))

# Define a function for text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Apply preprocessing to the sentences
df['processed_sentence'] = df['Sentences'].apply(preprocess_text)

# Encode the 'processed_sentence' column using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_sentence']).toarray()

# Label encode the 'Figure of Speech' column
le = LabelEncoder()
y = le.fit_transform(df['Figure_OF_Speech'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Neural Network Accuracy: {accuracy:.2f}")
print("Test Loss: ", loss)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.show()


# Function to predict figure of speech for a given sentence
def predict_figure_of_speech(sentence):
    processed_sentence = preprocess_text(sentence)
    processed_vector = vectorizer.transform([processed_sentence]).toarray()
    prediction = model.predict(processed_vector)
    predicted_label = le.inverse_transform([prediction.argmax()])[0]
    return predicted_label

# prediction
user_input = input("Write you sentence here: ")
predicted_figure_of_speech = predict_figure_of_speech(user_input)
print(f"The figure of speech in '{user_input}' is: {predicted_figure_of_speech}")

# Plot Bar Graph of Figure of Speech counts
fig, ax = plt.subplots()
df['Figure_OF_Speech'].value_counts().plot(kind='bar', ax=ax)
ax.set_title("Bar Graph of Figure of Speech counts")
ax.set_xlabel("Figure of Speech")
ax.set_ylabel("Count")
plt.show()

# Plot Pie Chart of Figure of Speech counts
fig, ax = plt.subplots()
df['Figure_OF_Speech'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
ax.set_ylabel('')
ax.set_title("Pie Chart of Figure of Speech counts")
plt.show()
