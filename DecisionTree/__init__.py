# Decision Trees are a classical approach of ML sort of like a seires of threshold based questions about the input features.
# Their main use case is to perform classification adn regression tasks.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_digits() #import dataset

X, y = data.data, data.target #set data params

#init train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= .20,
    random_state=42
)

#init scalar NOT NEEDED FOR DECSIONS TREES SINCE TRAINING DOESNT INVOLVE GRADIENT DESCNET
scaler = StandardScaler()


#scale data 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#The model can as most ask 5 sequential questions before making a predictions
#this is to prevent overfitting because these modesl tend to overfit. 
#Gini is the formula that decides which feature and threshold splits at each node. 
#Can also use, Entropy, Log Loss, MSE, Poisson.
tree = DecisionTreeClassifier(max_depth=5, criterion='gini')




#This is the training algorithm
#Looks at all features and possible threshholds then 
#calculates the Gini impurity reduction for every split
#it then picks th best one and splits data into two child nodes
#Repeated recursively untill max deph is 5
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
