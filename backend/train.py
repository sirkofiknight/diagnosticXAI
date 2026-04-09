import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

FEATURE_NAMES = ['Hours Studied', 'Attendance (%)', 'Sleep Hours']

if __name__ == '__main__':
    np.random.seed(42)
    n = 1500
    hours      = np.random.uniform(0, 10, n)
    attendance = np.random.uniform(0, 100, n)
    sleep      = np.random.uniform(0, 12, n)
    logit      = 1.2 * hours + 0.1 * attendance + 0.8 * sleep - 18 + np.random.normal(0, 1.5, n)
    passed     = (logit > 0).astype(int)

    X      = np.column_stack((hours, attendance, sleep))
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_sc, passed)

    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_names': FEATURE_NAMES}, f)

    print('Logistic Regression trained -> model.pkl')
