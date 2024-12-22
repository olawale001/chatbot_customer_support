import json
import numpy as np
from sklearn import metrics
import tensorflow as tf
import random
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

nltk.download("punkt")
nltk.download("wordnet")

with open('intents.json', 'r') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()    
words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['pattern']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = documents[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)    
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)    
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=8)
model.save('chatbot_model.h5')


pickle.dump(words, open('words.plk', 'wb'))
pickle.dump(classes, open('classes.plk', 'wb'))