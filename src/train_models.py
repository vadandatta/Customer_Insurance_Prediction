from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_all_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=0),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', random_state=0),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models