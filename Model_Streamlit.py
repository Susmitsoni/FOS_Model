import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import time as t

# CSS 
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf, #2e7bcf);
            color: white;
    }
    .stApp {
        background-color: #FFC0CB;
    }
    .classification-report-heading {
        font-size: 32px;
        font-weight: bold;
        color: black;
    }
    .classification-report {
        background-color: #D3D3D3;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv('FOS_CSV.csv', encoding='latin1')

stop_words = set(stopwords.words('english'))

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token is not None and token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

df['processed_sentence'] = df['Sentences'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_sentence']).toarray()

le = LabelEncoder()
y = le.fit_transform(df['Figure_OF_Speech'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Neural Network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Calculating loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)

# Sidebar
st.sidebar.write("# Welcome to Figure of Speech Predictor app!!!\n")
st.sidebar.header("Model Metrics")
st.sidebar.write(f"Neural Network Accuracy: {accuracy:.2f}")
st.sidebar.write(f"Test Loss: {loss:.2f}")

models = {
    'Random Forest': RandomForestClassifier(),
    'BernoulliNB': BernoulliNB(),
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB()
}

for model_name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"{model_name} Accuracy: {accuracy:.2f}")

# Sidebar options for graph visualization
st.sidebar.write("Do you want to see the data used in the form of a graph?")
show_graph = st.sidebar.checkbox("Yes")

if show_graph:
    graph_type = st.sidebar.radio("Choose the type of graph:", ("Bar Graph", "Pie Chart"))

# Streamlit title
st.title("Figure of Speech Predictor")

# Asking the user 
st.write("Do you know what a figure of speech is?")
if st.button("Yes"):
    st.write("Very well then.")
elif st.button("No"):
    st.write("A figure of speech is a rhetorical device that achieves a special effect by using words in distinctive ways. Examples include metaphor, simile, and hyperbole.")
    st.write("1. **Metaphor**: A figure of speech that directly compares two unlike things (e.g., 'Time is a thief').")
    st.write("2. **Simile**: A comparison using 'like' or 'as' (e.g., 'As brave as a lion').")
    st.write("3. **Hyperbole**: An exaggerated statement (e.g., 'I have a million things to do').")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# classification report 
st.markdown("<div class='classification-report-heading'>Classification Report:</div>", unsafe_allow_html=True)

report = classification_report(y_test, y_pred_classes, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()


st.dataframe(report_df.style.set_properties(**{'background-color': '#D3D3D3', 'color': 'black'}))

st.write("Do you know what a **classification report** is?")
if st.button("Yep"):
    st.write("Good! You know a lot.")
elif st.button("Nope"):
    st.write("A **classification report** is a summary of the performance of a classification model. It provides metrics that help assess how well the model predicts different classes.")
    st.write("**Precision** measures the proportion of true positive predictions (correctly predicted positive instances) out of all positive predictions (true positives + false positives).")
    st.write("**Recall** measures the proportion of true positive predictions out of all actual positive instances.")
    st.write("The **F1-score** is the harmonic mean of precision and recall. It balances both metrics.")
    st.write("**Support** represents the number of occurrences of each class in the true labels.")


st.write("\n\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# Function to predict figure of speech for a given sentence
def predict_figure_of_speech(sentence):
    processed_sentence = preprocess_text(sentence)
    processed_vector = vectorizer.transform([processed_sentence]).toarray()
    prediction = model.predict(processed_vector)
    predicted_label = le.inverse_transform([prediction.argmax()])[0]
    return predicted_label


# Streamlit App

st.header("Here you can find **Figure of Speech** present in different **Sentences**.")
st.info("To be on the safe side refrain from writing any numerical data")

st.write("Enter a paragraph to predict its figure of speech:")
user_input = st.text_area("Paragraph:")

if st.button("Predict"):
    with st.spinner("Please wait for some time"):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            t.sleep(0.04)  
            progress_bar.progress(percent_complete + 1)
    
    if user_input:
        sentences = sent_tokenize(user_input)
        predictions = []
        for i, sentence in enumerate(sentences):
            predicted_figure_of_speech = predict_figure_of_speech(sentence)
            predictions.append((i + 1, sentence, predicted_figure_of_speech))
        
        st.snow()  
        st.success("Prediction successful!")
        for i, sentence, predicted_figure_of_speech in predictions:
            st.write(f"Sentence {i}: '{sentence}' - Figure of Speech: {predicted_figure_of_speech}")
        
        st.warning("Remember this is a machine learning model, do not trust it completely.")
    else:
        st.write("Please enter a paragraph.")

# data visualization 
if show_graph:
    if graph_type == "Bar Graph":
        fig, ax = plt.subplots()
        df['Figure_OF_Speech'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Bar Graph of Figure of Speech counts")
        ax.set_xlabel("Figure of Speech")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    elif graph_type == "Pie Chart":
        fig, ax = plt.subplots()
        df['Figure_OF_Speech'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
        ax.set_ylabel('')
        ax.set_title("Pie Chart of Figure of Speech counts")
        st.pyplot(fig)
