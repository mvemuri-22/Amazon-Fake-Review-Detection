from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_and_split_data(df):
    # Load stop words and initialize stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Preprocessing function: remove stop words and apply stemming
    def preprocess_text(text):
        words = text.split()
        filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    # Split the data into training and testing sets (80% train, 20% test)
    X = df['text']  # Features (text data)
    y = df['label']  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply preprocessing to the training and testing sets separately
    X_train = X_train.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)

    # Tokenize and pad sequences
    max_words = 10000  # Hardcoded maximum number of words
    max_len = 100      # Hardcoded maximum sequence length
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Return the processed datasets and tokenizer
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer
