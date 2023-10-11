from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score

class SVM_classifier():

    def __init__(self):
        self.SVC = svm.SVC(kernel='rbf', C=1)
        

    def train(self, train_data, train_label):
        self.model = self.SVC.fit(train_data, train_label)
        
    def test(self, test_data, test_label):
        res = self.model.predict(test_data)
        acc = accuracy_score(test_label, res)
        return acc

    def all_score(self, test_data, test_label):
        res = self.model.predict(test_data)
        recall = recall_score(test_label, res, average=None)
        precision = precision_score(test_label, res, average=None)
        return recall, precision