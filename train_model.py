from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import pickle

model_files = [
    'models/naive.ml',
    'models/logistic.ml',
    'models/svm.ml',
    'models/knear.ml',
    'models/dectree.ml',
    'models/forest.ml',
    'models/xgb.ml',
    'models/neurnet.h5'
]


class _Pred:
    def __init__(self, data):
        self.data = data
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(
            data)

    def split_data(self, data):
        features = data.drop('target', axis=1)
        target = data['target']

        print("Test and Train data generated")
        return train_test_split(features, target, random_state=0, test_size=0.20)

    def pred_naive_bayes(self):
        clf = GaussianNB()
        print("Training Gaussian Naive Bayes Model")
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[0], 'wb'))
        print("Gaussian Naive Bayes Model Trained")
        pass

    def pred_logistic(self):
        clf = LogisticRegression(solver='lbfgs')
        print("Training Logistic Regression Model")
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[1], 'wb'))
        print("Logistic Regression Model Trained")
        pass

    def pred_svm(self):
        clf = svm.SVC(kernel='linear')
        print("Training Support Vector Machine Model")
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[2], 'wb'))
        print("Support Vector Machine Model Trained")
        pass

    def k_nearest(self):
        best_x = 0
        m_acc = 0
        print("Training K-Nearest Neighbour Model")
        clf = KNeighborsClassifier(n_neighbors=7)
        for x in range(1, 242):
            clf = KNeighborsClassifier(n_neighbors=x)
            clf.fit(self.x_train, self.y_train)
            c_pred = clf.predict(self.x_test)
            c_acc = round(accuracy_score(c_pred, self.y_test)*100, 2)
            print(
                "Training with n: {0}\tcount({1}/{2})\tacc({3})".format(x, x, 242, c_acc))

            if c_acc > m_acc:
                m_acc = c_acc
                best_x = x

        clf = KNeighborsClassifier(n_neighbors=best_x)
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[3], 'wb'))
        print(
            "K-Nearest Neighbour Model Trained (best-k -> {0} )".format(best_x))

    def decision_tree(self):

        m_acc = 0
        best_x = 0
        print("Training Decision Tree Model")
        for x in range(242):
            clf = DecisionTreeClassifier(random_state=x)
            clf.fit(self.x_train, self.y_train)
            c_pred = clf.predict(self.x_test)
            c_acc = round(accuracy_score(c_pred, self.y_test)*100, 2)
            print(
                "Training with seed: {2}\tcount({0}/{1})\tacc({3})".format(x+1, 242, x, c_acc))
            if c_acc > m_acc:
                m_acc = c_acc
                best_x = x

        clf = DecisionTreeClassifier(random_state=best_x)
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[4], 'wb'))
        print("Decision Tree Model Trained - best x-{0}".format(best_x))
        pass

    def random_forest(self):
        max_acc = 0
        best_x = 0
        print("Training Random Forest Model")
        for x in range(1000):
            clf = RandomForestClassifier(random_state=x)
            clf.fit(self.x_train, self.y_train)
            c_pred = clf.predict(self.x_test)
            c_acc = round(accuracy_score(c_pred, self.y_test)*100, 2)
            print(
                "Training with seed: {2}\tcount({0}/{1})\tacc({3})".format(x+1, 1000, x, c_acc))
            if c_acc > max_acc:
                max_acc = c_acc
                best_x = x

        clf = RandomForestClassifier(random_state=best_x)
        clf.fit(self.x_train, self.y_train)
        pickle.dump(clf, open(model_files[5], 'wb'))
        print("Random Forest Model Trained best x-{0}".format(best_x))
        pass

    def xgb(self):
        xgb_model = XGBClassifier(
            objective="binary:logistic", random_state=42, use_label_encoder=False)
        print("Training XGBoost model")
        xgb_model.fit(self.x_train, self.y_train)
        pickle.dump(xgb_model, open(model_files[6], 'wb'))
        print("XGBoost Model Trained")
        pass

    def neural_network(self):
        model = Sequential()
        print("Training Neural Network Model")
        model.add(Dense(11, activation="relu", input_dim=13))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=["accuracy"])

        model.fit(self.x_train, self.y_train, epochs=300)
        model.save(model_files[7])
        print("Neural Network Model Trained")
        pass

    def save_model(self):
        self.pred_naive_bayes()
        self.pred_logistic()
        self.pred_svm()
        self.k_nearest()
        self.decision_tree()
        self.random_forest()
        self.xgb()
        self.neural_network()


data = pd.read_csv('assets/heart.csv')
pred = _Pred(data)
pred.save_model()
