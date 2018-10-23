# Q5

y_proba_lr = m.fit(X_train, y_train).predict_proba(X_test)

precision, recall, thresholds = precision_recall_curve(y_test, y_proba_lr[0::,1])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

ind08, idx = find_nearest(recall, 0.8)
prec08 = precision[idx]
print(prec08)
# Answer = 0.55
# plt.figure()
# plt.xlim([0.0, 1.01])
# plt.ylim([0.0, 1.01])
# plt.plot(precision, recall, label='Precision-Recall Curve')
# plt.xlabel('Precision', fontsize=16)
# plt.ylabel('Recall', fontsize=16)
# plt.axes().set_aspect('equal')
# plt.show()

# Q8
print(m)
from sklearn.metrics import classification_report
m_predicted_mc = m.predict(X_test)
print(precision_score(y_test, m_predicted_mc, average = 'micro'))
# Answer - 0.744

# Q13
print(m)
grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
grid_clf_acc = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)
y_predicted = grid_clf_acc.predict(X_test)
prec = precision_score(y_test, y_predicted)
rec = recall_score(y_test, y_predicted)
print(rec - prec)
# Answer - 0.52

# Q14
print(m)
grid_values = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
grid_clf_acc = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_clf_acc.fit(X_train, y_train)
y_predicted = grid_clf_acc.predict(X_test)
prec = precision_score(y_test, y_predicted)
rec = recall_score(y_test, y_predicted)
print(rec - prec)
# Answer - 0.15