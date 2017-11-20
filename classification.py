import numpy as np

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

def main():

    train_data = np.loadtxt('train_data.csv' , delimiter=',',)
    test_data = np.loadtxt('test_data.csv' , delimiter=',',)

    print "TRAIN DATA"
    print train_data.shape

    print "TEST DATA"
    print test_data.shape

    train_X = train_data[:,:-1]
    train_Y = train_data[:,-1]

    test_X = test_data[:,:-1]
    test_Y = test_data[:,-1]

    # FOR CROSS VALIDATION WITH SPLIT 40%
    X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.4, random_state=0)

    #############################################################################################
    
    classifiers = [ 
                    DecisionTreeClassifier(),
                    KNeighborsClassifier(n_neighbors=9),
                    #GaussianNB(),
                    #MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=3),
                    #NearestCentroid(),
                    RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs = -1),
                    SVC(decision_function_shape='ovo'), 
                    #SVC(kernel='linear', C=1),
                  ]

    for clf in classifiers:
        print ""
        print clf
        #print clf.classes_

        #clf.fit(train_X, train_Y)
        #print(confusion_matrix(test_Y, clf.predict(test_X)))                                                    
        
        #print(clf.score(test_X, test_Y)) 
        #clf.fit(X_train, Y_train)
        #print(clf.score(X_test, Y_test))                                              
        #print(confusion_matrix(Y_test, clf.predict(X_test))) 
 
        predicted = cross_val_predict(clf, train_X, train_Y, cv=10)
        print("Cross-validation accuracy: ", metrics.accuracy_score(train_Y, predicted))

if __name__ == "__main__":
    main()