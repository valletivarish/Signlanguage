import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Set the desired sequence length
desired_length = 84  # Adjust this to your desired length

# Pad or truncate sequences to the desired length
data_padded = [np.pad(sample, (0, desired_length - len(sample)), mode='constant') if len(sample) < desired_length else sample[:desired_length] for sample in data_dict['data']]

# Create a NumPy array with dtype=object
data = np.array(data_padded, dtype=object)

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
