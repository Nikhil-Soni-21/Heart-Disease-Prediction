from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from keras.models import load_model
import pickle

model_files = [
    'models/naive.mod',
    'models/logistic.mod',
    'models/svm.mod',
    'models/knear.mod',
    'models/dectree.mod',
    'models/forest.mod',
    'models/xgb.mod',
    'models/neurnet.h5'
]


class ModelLoader:
    def __init__(self, data):
        self.data = data
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(
            data)

    def split_data(self, data):
        features = data.drop('target', axis=1)
        target = data['target']
        return train_test_split(features, target, random_state=0, test_size=0.20)

    def model(self, index):
        model = pickle.load(open(model_files[index], 'rb'))
        return model.predict(self.x_test)

    def neunet(self):
        model = load_model(model_files[7])
        y_pred = model.predict(self.x_test)
        rounded = [round(x[0]) for x in y_pred]
        return rounded
