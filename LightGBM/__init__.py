import lightgbm as lgb

# --- LightGBM: Scikit-learn API ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=-1,            # -1 means no limit (leaf-wise handles this)
    num_leaves=31,           # KEY PARAM: controls complexity in leaf-wise growth
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,    # minimum data in a leaf (prevents overfitting)
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

lgb_preds = lgb_model.predict(X_test)
print("\n" + "=" * 60)
print("LIGHTGBM RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, lgb_preds):.4f}")

