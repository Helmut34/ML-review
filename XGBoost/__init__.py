import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=5000, n_features=20, n_informative=10, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,

    test_size=0.2,
    random_state=42


)


xgb_model = xgb.XGBClassifier(
    n_estimators=200,       #Number of Trees
    max_depth=6,             #max deph per tree to control overfitting.
    learning_rate=0.1,      
    subsample=0.8,           #Fractions of sample tper trre
    colsample_bytree=0.8,       #Fractuibs if features per tree
    reg_alpha=0.1,              #L1 Regularization
    reg_lambda=1.0,             #L2 Regularization
    eval_metric='logloss',      
    random_state=42
)


xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # monitor performance on test set
    verbose=False
)

xgb_preds = xgb_model.predict(X_test)
print("=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"Feature importances (top 5):")
importances = xgb_model.feature_importances_
top5 = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)[:5]
for idx, imp in top5:
    print(f"  Feature {idx}: {imp:.4f}")