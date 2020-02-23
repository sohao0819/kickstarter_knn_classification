import pickle
import pandas as pd
from model import KickstarterModel

df = pd.read_csv("data/train.zip")

my_model = KickstarterModel()
X_train, y_train = my_model.preprocess_training_data(df)
my_model.fit(X_train, y_train)

# Save to pickle
with open('model.pickle', 'wb') as f:
    pickle.dump(my_model, f)
