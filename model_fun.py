from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split_data(data):
    features = data.drop('target', axis=1)
    target = data['target']
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, random_state=0, test_size=0.20)
    return (x_train, x_test, y_train, y_test)
    pass


def pred_naive_bayes(data):
    frame = split_data(data)
    clf = GaussianNB()
    clf.fit(frame[0], frame[2])
    y_pred = clf.predict(frame[1])
    return accuracy_score(y_pred, frame[3])
