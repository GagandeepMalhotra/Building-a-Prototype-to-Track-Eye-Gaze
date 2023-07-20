import matplotlib.pyplot as plt
import pickle
import os
import cv2
import numpy as np
import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

def get_x_y(df):
    x = []
    y = []
    for name in df['name']:
        loaded_npz = np.load(name)

        pic = loaded_npz['pic_data']
        x.append(pic)

        vector_x, vector_y, vector_z = loaded_npz['vector_in']
        y.append([vector_x, vector_y, vector_z])

    x = np.array(x)
    y = np.array(y)
    return x, y

def main():
    #names_df = load_pkl_dataset()
    names_df = load_npz_dataset()
    shuffled_df = names_df.sample(frac=1)

    train_df, val_df, test_df = create_dataframes(shuffled_df)
    X_test, y_test = get_x_y(test_df)
    
    X_train, y_train = get_x_y(train_df)
    X_val, y_val = get_x_y(val_df)

    #grid_result = get_grid_search(X_train, y_train, X_val, y_val)

    model = create_model(X_train, y_train, X_val, y_val)

    output_df = get_load_model(test_df, X_test)
    avg_xyz = get_average_difference(output_df)
    
    prediction = get_prediction(r'SynthEyes_data\f01\f01_1002_0.1963_0.3927.npz')

def get_grid_search(X_train, y_train, X_val, y_val):
    model = KerasRegressor(build_fn=create_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, verbose=1)

    #Define the grid search parameters
    batch_size = [8, 16]
    epochs = [50]
    param_grid = dict(batch_size=batch_size, epochs=epochs)

    #Perform the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    #Print the best parameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result

def get_prediction(npz_file_path):
    model = models.load_model('model3/')
    npz_file = np.load(npz_file_path)
    pic_data = npz_file['pic_data'].reshape((1, 80, 120, 3))
    prediction = model.predict(pic_data)
    print("predicted look_vector: ", prediction)
    return prediction

def compute_difference(row):
    x = np.array([row['x'], row['y'], row['z']])
    pred = np.array(row['Predicted look_vector'])
    abs_diff = np.abs(x - pred)
    return abs_diff

def get_load_model(df, input_pic):
    loaded_model = models.load_model('model3/')

    #Use the loaded model to obtain predictions on the test set
    test_predictions = loaded_model.predict(input_pic)
    #test_predictions = loaded_model.predict(input_pic).flatten()

    test_preds_series = pd.Series(test_predictions.tolist())
    test_preds_series.hist()
    plt.show()

    df.reset_index(drop=True, inplace=True)
    df['Predicted look_vector'] = test_preds_series
    df['difference'] = df.apply(compute_difference, axis=1)

    print(tabulate(df, headers='keys', tablefmt='psql', showindex='always'))
    return df

def get_average_difference(df):
    xyz_values = [np.array(row) for row in df['difference']]
    avg_xyz = np.mean(xyz_values, axis=0)
    print("Avg_XYZ: ", avg_xyz)
    return avg_xyz

def create_model(X_train, y_train, X_val, y_val):
    from keras import backend as K

    def custom_activation(x):
        return K.clip(x, -0.1, 0.1)
    
    #Define the CNN architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation=custom_activation))

    #Compile the model with an appropriate loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='Adam')

    cp = ModelCheckpoint('model3/', monitor='val_loss', mode='min', save_best_only=True)

    #Train the model on the dataset
    model.fit(X_train, y_train, epochs=5, batch_size=8, validation_data=(X_val, y_val), callbacks=[cp])
    return model

def get_loss(model, X_test, y_test):
    #Evaluate the model on a separate test dataset
    loss = model.evaluate(X_test, y_test)
    print(loss)
    return loss

def get_all_vectors(names_df):
    df = names_df[['x','y','z']]
    return df

def drop_columns(df, column_list):
    df.drop(column_list, inplace=True, axis=1)
    return df
    
def load_npz_dataset():
    npz_paths = glob.glob('SynthEyes_data/**/*.npz', recursive=True)

    #data_list = [(name,) for name in npz_paths]
    #names_df = pd.DataFrame(data_list, columns=['name', 'x'])
    
    data_list = []
    for name in npz_paths:
        with np.load(name) as data:
            x, y, z = data['vector_in'] 
            data_list.append((name, x, y, z))

    names_df = pd.DataFrame(data_list, columns=['name', 'x', 'y', 'z'])
    return names_df

def create_dataframes(df):
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_df, val_df, test_df = df[:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]
    return train_df, val_df, test_df

def load_pkl_dataset():
    images = np.array([])
    look_vector_x = np.array([])
    look_vector_y = np.array([])
    look_vector_z = np.array([])
    npz_paths = glob.glob('SynthEyes_data/**/*.pkl', recursive=True)

    for name in npz_paths:
        image_file_path, file_look_vector_x, file_look_vector_y, file_look_vector_z = open_file(name)

        images = np.append(images, image_file_path)
        look_vector_x = np.append(look_vector_x, file_look_vector_x)
        look_vector_y = np.append(look_vector_y, file_look_vector_y)
        look_vector_z = np.append(look_vector_z, file_look_vector_z)

    names_df = pd.DataFrame({'name':[y for y in images],
                             'x':[x for x in look_vector_x],
                             'y':[y for y in look_vector_y],
                             'z':[z for z in look_vector_z]})

    npz_paths = []
    for i, row in names_df.iterrows():
        picture_path = row['name']
        npz_path = picture_path[:-4] + '.npz'
        npz_paths.append(npz_path)

        pic_bgr_arr = cv2.imread(picture_path)
        pic_rgb_arr = cv2.cvtColor(pic_bgr_arr, cv2.COLOR_BGR2RGB)

        vector_in = np.array([row['x'], row['y'], row['z']])

        np.savez_compressed(npz_path, pic_data=pic_rgb_arr, vector_in=vector_in)
    
    names_df['npz_path'] = pd.Series(npz_paths)
    return names_df

def open_file(file_name):
    #Open the file in binary mode
    with open(file_name, 'rb') as f:
        #Load the pickled object
        data = pickle.load(f)
        image_file_path = file_name.replace(".pkl", ".png")
        file_look_vectors = [v for k, v in data.items() if k == 'look_vec'][0]
        look_vector_x = file_look_vectors[0]
        look_vector_y = file_look_vectors[1]
        look_vector_z = file_look_vectors[2]
    return image_file_path, look_vector_x, look_vector_y, look_vector_z

if __name__ == '__main__':
    #Profile.run("main()")
    #main()
    pass

