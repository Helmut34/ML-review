from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_digits()

X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(

    X, y,
    test_size=.20,
    random_state=42
) 


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

forest = RandomForestClassifier(
    n_estimators=100, #number of trees
    max_depth=5, #deph of each tree
    max_features='sqrt', #Only considser sqrt(n features) per split
    random_state=42

)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")