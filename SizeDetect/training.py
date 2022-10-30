import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

name_sz = 'size20'
name_not_xz = 'error_size20'
name_model = "model20.json"
name_h5 = "model20.h5"

data_0 = pd.read_excel(r'size.xlsx', sheet_name=name_not_xz)
X1_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d1'])
X2_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d2'])
X3_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d3'])
X_0 = np.concatenate((X1_0, X2_0, X3_0), axis=1)
Y_0 = np.array(pd.DataFrame(data_0.iloc[0:30, :], columns=['y']))

# true size or not
data = pd.read_excel(r'size.xlsx', sheet_name=name_sz)
X1 = pd.DataFrame(data.iloc[0:15, :], columns=['d1'])
X2 = pd.DataFrame(data.iloc[0:15, :], columns=['d2'])
X3 = pd.DataFrame(data.iloc[0:15, :], columns=['d3'])
X = np.concatenate((X1, X2, X3), axis=1)
Y = np.array(pd.DataFrame(data.iloc[0:15, :], columns=['y']))

X_train = sc.fit_transform(np.concatenate((X, X_0), axis=0))
Y_train = np.concatenate((Y, Y_0), axis=0)

# create model
classifier = Sequential()
classifier.add(Dropout(1/3, input_shape=(3,)))
classifier.add(Dense(units=10, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=10, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile model
sgd = SGD(learning_rate=0.2, momentum=0.9, decay=0.0005)
classifier.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
classifier.fit(X_train, Y_train, batch_size=15, epochs=200)

# evaluate the model
scores = classifier.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1] * 100))

# serialize model to JSON
model_json = classifier.to_json()
with open(name_model, "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
classifier.save_weights(name_h5)
print("Saved model to disk")

print("\n")
y_pred = classifier.predict(X_train)
print(y_pred)
