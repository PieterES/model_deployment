import pickle
from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_features=round(len(X_train.columns)/3),
        max_depth=len(X_train.columns)
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
