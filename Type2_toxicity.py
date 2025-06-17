
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
tf.compat.v1.reset_default_graph()

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppresses INFO and WARNING messages from TensorFlow
warnings.filterwarnings("ignore", category=UserWarning)



st.title("🧠 Toxic Comment Classification App")

st.markdown("""
This app allows you to **train a model** on a toxic comments dataset and test it on your own inputs.
""")

# Load data
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Test", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
   

# ✅ Convert label columns to numeric
    # Step 1: Select text column
    text_column = st.selectbox(
    "Select the text column:",
    df.select_dtypes(include=['object']).columns.tolist(),
    index=df.columns.get_loc("text_comment") if "text_comment" in df.columns else 0)


# Step 2: Restrict label selection to numeric columns only
    numeric_cols = df.select_dtypes(include=['int', 'float', 'bool']).columns.tolist()
    label_columns = st.multiselect(
    "Select label columns (multi-label classification):",
    options=numeric_cols,
    default=[col for col in numeric_cols if col.lower() in ["toxic", "insult", "obscene"]] ) # You can adjust this default list


    # Show column options
    #text_column = st.selectbox("Select the text column:", df.columns)
    #text_column = st.selectbox("Select the text column:", df.columns, index=df.columns.get_loc("text_comment") if "text_comment" in df.columns else 0)
    #label_columns = st.multiselect("Select label columns (multi-label classification):", df.columns, default=[col for col in df.columns if col != text_column])
    
    if text_column and label_columns:
        # Text cleaning
        def clean_text(text):
            text = text.lower()
            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
            text = re.sub(r"\w*\d\w*", "", text)
            text = re.sub(r" +", " ", text)
            return text

        df[text_column] = df[text_column].astype(str).apply(clean_text)

        # Tokenization and padding
        tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
        tokenizer.fit_on_texts(df[text_column])
        sequences = tokenizer.texts_to_sequences(df[text_column])
        max_len = 200
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')
        X = padded
        #y = df[label_columns].values
        y = df[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ Ensure correct dtype for training
        #y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
        #X = np.array(padded)
        #y = df[label_columns].values

        #X = padded
        #y = df[label_columns].values

        # Train-test split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.header("Step 2: Train Model")
        if st.button("Train LSTM Model"):
            model = Sequential()
            model.add(Embedding(input_dim=20000, output_dim=128, input_length=max_len))
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(32)))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(label_columns), activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

            st.success("✅ Training complete!")

            # Plotting
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].plot(history.history['accuracy'], label='Train Acc')
            ax[0].plot(history.history['val_accuracy'], label='Val Acc')
            ax[0].legend()
            ax[0].set_title('Accuracy')

            ax[1].plot(history.history['loss'], label='Train Loss')
            ax[1].plot(history.history['val_loss'], label='Val Loss')
            ax[1].legend()
            ax[1].set_title('Loss')

            st.pyplot(fig)
            # Step 3: Predict
            st.header("Step 3: Test the Model")
            user_input = st.text_area("Enter a comment:")

            if user_input:
                if 'model' in st.session_state and 'tokenizer' in st.session_state:
                    model = st.session_state['model']
                    tokenizer = st.session_state['tokenizer']
        
                    cleaned = clean_text(user_input)
                    seq = tokenizer.texts_to_sequences([cleaned])
                    pad = pad_sequences(seq, maxlen=max_len, padding='post')
                    preds = model.predict(pad)[0]

                    pred_df = pd.DataFrame({'Label': label_columns, 'Probability': preds})
                    pred_df['Toxic?'] = pred_df['Probability'] > 0.5
                    st.write(pred_df)
            else:
                st.warning("⚠️ Please train the model first.")

            
            
