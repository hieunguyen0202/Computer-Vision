from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

sc = StandardScaler()

choose = 16
valid = [[414.566038165212, 230.7834482799839, 248.9738942138312]]

if choose == 16:
    # load json and create model
    json_file = open('model16.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model16.h5")
    print("Loaded model from disk")

    data_0 = pd.read_excel(r'size.xlsx', sheet_name='error_size16')
    X1_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d1'])
    X2_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d2'])
    X3_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d3'])
    X_0 = np.concatenate((X1_0, X2_0, X3_0), axis=1)

    # size 16 or not
    data_16 = pd.read_excel(r'size.xlsx', sheet_name='size16')
    X1_16 = pd.DataFrame(data_16.iloc[0:15, :], columns=['d1'])
    X2_16 = pd.DataFrame(data_16.iloc[0:15, :], columns=['d2'])
    X3_16 = pd.DataFrame(data_16.iloc[0:15, :], columns=['d3'])
    X_16 = np.concatenate((X1_16, X2_16, X3_16), axis=1)

    X16_0_train = sc.fit(np.concatenate((X_16, X_0), axis=0))

    y_pred = loaded_model.predict(sc.transform(valid))
    print(y_pred, y_pred > 0.5)

elif choose == 18:
    # load json and create model
    json_file = open('model18.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model18.h5")
    print("Loaded model from disk")

    data_0 = pd.read_excel(r'size.xlsx', sheet_name='error_size18')
    X1_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d1'])
    X2_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d2'])
    X3_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d3'])
    X_0 = np.concatenate((X1_0, X2_0, X3_0), axis=1)

    # size 18 or not
    data_18 = pd.read_excel(r'size.xlsx', sheet_name='size18')
    X1_18 = pd.DataFrame(data_18.iloc[0:15, :], columns=['d1'])
    X2_18 = pd.DataFrame(data_18.iloc[0:15, :], columns=['d2'])
    X3_18 = pd.DataFrame(data_18.iloc[0:15, :], columns=['d3'])
    X_18 = np.concatenate((X1_18, X2_18, X3_18), axis=1)

    X18_0_train = sc.fit(np.concatenate((X_18, X_0), axis=0))

    y_pred = loaded_model.predict(sc.transform(valid))
    print(y_pred, y_pred > 0.5)

elif choose == 20:
    # load json and create model
    json_file = open('model20.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model20.h5")
    print("Loaded model from disk")

    data_0 = pd.read_excel(r'size.xlsx', sheet_name='error_size20')
    X1_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d1'])
    X2_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d2'])
    X3_0 = pd.DataFrame(data_0.iloc[0:30, :], columns=['d3'])
    X_0 = np.concatenate((X1_0, X2_0, X3_0), axis=1)

    # size 20 or not
    data_20 = pd.read_excel(r'size.xlsx', sheet_name='size20')
    X1_20 = pd.DataFrame(data_20.iloc[0:15, :], columns=['d1'])
    X2_20 = pd.DataFrame(data_20.iloc[0:15, :], columns=['d2'])
    X3_20 = pd.DataFrame(data_20.iloc[0:15, :], columns=['d3'])
    X_20 = np.concatenate((X1_20, X2_20, X3_20), axis=1)

    X20_0_train = sc.fit(np.concatenate((X_20, X_0), axis=0))

    y_pred = loaded_model.predict(sc.transform(valid))
    print(y_pred, y_pred > 0.5)
