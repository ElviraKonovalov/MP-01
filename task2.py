import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

'''
Q2 - Load the dataset
'''
df = pd.read_csv('drug200.csv')

'''
Q3 - Plot the distribution of the instances in each class and store the graphic
'''
classes = np.array(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
class_instance_count = np.zeros_like(classes)

for i in range(classes.shape[0]):
	class_instance_count[i] = (df.Drug.values == classes[i]).sum()

plt.bar(classes,class_instance_count)
plt.title('Drug Distribution')
plt.xlabel('Class Name')
plt.ylabel('Number of Instances')
plt.savefig("drug-distribution.pdf")

'''
Q4 - Convert all ordinal and nominal features in numerical format
'''
age = df['Age']
sex = np.array(pd.get_dummies(df['Sex'], drop_first=True)).reshape(-1,) # 1=M, 0=F
bp = pd.Categorical(df['BP'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
cholesterol = pd.Categorical(df['Cholesterol'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
# bp = pd.Categorical(df['BP'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH'])
# cholesterol = pd.Categorical(df['Cholesterol'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH'])
na_to_k = df['Na_to_K']


X = np.column_stack((age, sex, bp, cholesterol, na_to_k))
y = df['Drug']

'''
Q5 - Split the dataset using the default parameter values
'''
X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
Q6 - Run 6 different classifiers:
'''

# (a) NB: a Gaussian Naive Bayes Classifier with the default parameters
nb = GaussianNB().fit(X_train, y_train) # Create a model using training data
nb_y_test_pred = nb.predict(X_test) # Use the model created to make predictions on the test data

# (b) Base-DT: a Decision Tree with the default parameters
base_dt = DecisionTreeClassifier().fit(X_train, y_train)
base_dt_y_test_pred = base_dt.predict(X_test) # Use the model created to make predictions on the test data

# (c) Top-DT: a better performing Decision Tree
parameters = {'criterion':('gini', 'entropy'), 'max_depth':[3, 10], 'min_samples_split':[3, 5, 6]}
top_dt = GridSearchCV(base_dt, parameters).fit(X_train, y_train)
top_dt_y_test_pred = top_dt.predict(X_test) # Use the model created to make predictions on the test data

# (d) PER: a Perceptron, with default parameter values
perceptron = Perceptron().fit(X_train, y_train)
perceptron_y_test_pred = perceptron.predict(X_test) # Use the model created to make predictions on the test data

# (e) Base-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
base_mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd', max_iter=5000).fit(X_train, y_train)
base_mlp_y_test_pred = base_mlp.predict(X_test) # Use the model created to make predictions on the test data

# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search
parameters = {'activation':('logistic', 'tanh', 'relu', 'identity'), 'hidden_layer_sizes':[(30, 50), (10, 10, 10)], 'solver':('adam', 'sgd')}
top_mlp = GridSearchCV(base_mlp, parameters).fit(X_train, y_train)
top_mlp_y_test_pred = top_mlp.predict(X_test) # Use the model created to make predictions on the test data

'''
Q7 - Metric evaluation
'''
with open('drugs-performance.txt', 'a') as f:
	'''
	NB: a Gaussian Naive Bayes Classifier with the default parameters
	'''
	# (a)
	a = "\n\n************************\n\n(a)\nNB: a Gaussian Naive Bayes Classifier with the default parameters"
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, nb_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, nb_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, nb_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, nb_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, nb_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	'''
	Base-DT: a Decision Tree with the default parameters
	'''
	# (a)
	a = "\n\n************************\n\n(a)\nBase-DT: a Decision Tree with the default parameters"
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, base_dt_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, base_dt_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, base_dt_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_dt_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_dt_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	'''
	Top-DT: a better performing Decision Tree
	'''
	# (a)
	best_parameters = top_dt.best_params_
	a = "\n\n************************\n\n(a)\n'Top-DT' a better performing Decision Tree with the following hyper-parameters:\n" + str(best_parameters)
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, top_dt_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, top_dt_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, top_dt_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_dt_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_dt_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	'''
	PER: a Perceptron, with default parameter values
	'''
	# (a)
	a = "\n\n************************\n\n(a)\nPER: a Perceptron, with default parameter values"
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, perceptron_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, perceptron_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, perceptron_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, perceptron_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, perceptron_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	'''
	Base-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
	'''
	# (a)
	a = "\n\n************************\n\n(a)\nBase-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.\nAlso adjusted max_iter=5000 to converge."
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, base_mlp_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, base_mlp_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, base_mlp_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_mlp_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_mlp_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	'''
	Top-MLP: a better performing Multi-Layered Perceptron found using grid search
	'''
	# (a)
	best_parameters = top_mlp.best_params_
	a = "\n\n************************\n\n(a)\n'Top-MLP' a better performing Multi-Layered Perceptron found using grid search with the following hyper-parameters:\n" + str(best_parameters)
	f.writelines(a)

	# (b) The confusion matrix
	b = confusion_matrix(y_test, top_mlp_y_test_pred)
	f.writelines("\n\n(b)\n"+str(b))

	# (c) The precision, recall, and F1-measure for each class
	c = classification_report(y_test, top_mlp_y_test_pred)
	f.writelines("\n\n(c)\n"+str(c))

	# (d) The accuracy, macro-average F1 and weighted-average F1 of the model
	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, top_mlp_y_test_pred))
	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_mlp_y_test_pred, average='macro'))
	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_mlp_y_test_pred, average='weighted'))
	f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

'''
Q8 - 
'''
avg_accuracy, avg_macro_f1, avg_weighted_f1, accuracy_sd, macro_f1_sd, weighted_f1_sd   = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
nb_accuracies, nb_macro_f1s, nb_weighted_f1s, nb_precisions, nb_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
base_dt_accuracies, base_dt_macro_f1s, base_dt_weighted_f1s, base_dt_precisions, base_dt_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
top_dt_accuracies, top_dt_macro_f1s, top_dt_weighted_f1s, top_dt_precisions, top_dt_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
perceptron_accuracies, perceptron_macro_f1s, perceptron_weighted_f1s, perceptron_precisions, perceptron_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
base_mlp_accuracies, base_mlp_macro_f1s, base_mlp_weighted_f1s, base_mlp_precisions, base_mlp_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)
top_mlp_accuracies, top_mlp_macro_f1s, top_mlp_weighted_f1s, top_mlp_precisions, top_mlp_recalls = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)

for i in range(10):
    # (a) NB: a Gaussian Naive Bayes Classifier with the default parameters
    nb = GaussianNB().fit(X_train, y_train) # Create a model using training data
    nb_y_test_pred = nb.predict(X_test) # Use the model created to make predictions on the test data

    accuracy = accuracy_score(y_test, nb_y_test_pred)
    precision = precision_score(y_test, nb_y_test_pred, average='micro')
    recall = recall_score(y_test, nb_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, nb_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, nb_y_test_pred, average='weighted')

    nb_accuracies[i] = accuracy
    nb_precisions[i] = precision
    nb_recalls[i] = recall
    nb_macro_f1s[i] = f1_macro
    nb_weighted_f1s[i] = f1_weighted
    
    # (b) Base-DT: a Decision Tree with the default parameters
    base_dt = DecisionTreeClassifier().fit(X_train, y_train)
    base_dt_y_test_pred = base_dt.predict(X_test) # Use the model created to make predictions on the test data
    
    accuracy = accuracy_score(y_test, base_dt_y_test_pred)
    precision = precision_score(y_test, base_dt_y_test_pred, average='micro')
    recall = recall_score(y_test, base_dt_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, base_dt_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, base_dt_y_test_pred, average='weighted')

    base_dt_accuracies[i] = accuracy
    base_dt_precisions[i] = precision
    base_dt_recalls[i] = recall
    base_dt_macro_f1s[i] = f1_macro
    base_dt_weighted_f1s[i] = f1_weighted
    
    # (c) Top-DT: a better performing Decision Tree
    parameters = {'criterion':('gini', 'entropy'), 'max_depth':[3, 10], 'min_samples_split':[3, 5, 6]}
    top_dt = GridSearchCV(base_dt, parameters).fit(X_train, y_train)
    top_dt_y_test_pred = top_dt.predict(X_test) # Use the model created to make predictions on the test data
    
    accuracy = accuracy_score(y_test, top_dt_y_test_pred)
    precision = precision_score(y_test, top_dt_y_test_pred, average='micro')
    recall = recall_score(y_test, top_dt_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, top_dt_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, top_dt_y_test_pred, average='weighted')

    top_dt_accuracies[i] = accuracy
    top_dt_precisions[i] = precision
    top_dt_recalls[i] = recall
    top_dt_macro_f1s[i] = f1_macro
    top_dt_weighted_f1s[i] = f1_weighted
    
    # (d) PER: a Perceptron, with default parameter values
    perceptron = Perceptron().fit(X_train, y_train)
    perceptron_y_test_pred = perceptron.predict(X_test) # Use the model created to make predictions on the test data
    
    accuracy = accuracy_score(y_test, perceptron_y_test_pred)
    precision = precision_score(y_test, perceptron_y_test_pred, average='micro')
    recall = recall_score(y_test, perceptron_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, perceptron_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, perceptron_y_test_pred, average='weighted')

    perceptron_accuracies[i] = accuracy
    perceptron_precisions[i] = precision
    perceptron_recalls[i] = recall
    perceptron_macro_f1s[i] = f1_macro
    perceptron_weighted_f1s[i] = f1_weighted
    
    # (e) Base-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
    base_mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd', max_iter=5000).fit(X_train, y_train)
    base_mlp_y_test_pred = base_mlp.predict(X_test) # Use the model created to make predictions on the test data
    
    accuracy = accuracy_score(y_test, base_mlp_y_test_pred)
    precision = precision_score(y_test, base_mlp_y_test_pred, average='micro')
    recall = recall_score(y_test, base_mlp_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, base_mlp_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, base_mlp_y_test_pred, average='weighted')

    base_mlp_accuracies[i] = accuracy
    base_mlp_precisions[i] = precision
    base_mlp_recalls[i] = recall
    base_mlp_macro_f1s[i] = f1_macro
    base_mlp_weighted_f1s[i] = f1_weighted
    
    # (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search
    parameters = {'activation':('logistic', 'tanh', 'relu', 'identity'), 'hidden_layer_sizes':[(30, 50), (10, 10, 10)], 'solver':('adam', 'sgd')}
    top_mlp = GridSearchCV(base_mlp, parameters).fit(X_train, y_train)
    top_mlp_y_test_pred = top_mlp.predict(X_test) # Use the model created to make predictions on the test data
    
    accuracy = accuracy_score(y_test, top_mlp_y_test_pred)
    precision = precision_score(y_test, top_mlp_y_test_pred, average='micro')
    recall = recall_score(y_test, top_mlp_y_test_pred, average='micro')
    f1_macro = f1_score(y_test, top_mlp_y_test_pred, average='macro')
    f1_weighted = f1_score(y_test, top_mlp_y_test_pred, average='weighted')

    top_mlp_accuracies[i] = accuracy
    top_mlp_precisions[i] = precision
    top_mlp_recalls[i] = recall
    top_mlp_macro_f1s[i] = f1_macro
    top_mlp_weighted_f1s[i] = f1_weighted

# avg_accuracy, avg_macro_f1, avg_weighted_f1, accuracy_sd, macro_f1_sd, weighted_f1_sd   = np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)
# for i in range(10):
# 	# (a) NB: a Gaussian Naive Bayes Classifier with the default parameters
# 	nb = GaussianNB().fit(X_train, y_train) # Create a model using training data
# 	nb_y_test_pred = nb.predict(X_test) # Use the model created to make predictions on the test data

# 	# (b) Base-DT: a Decision Tree with the default parameters
# 	base_dt = DecisionTreeClassifier().fit(X_train, y_train)
# 	base_dt_y_test_pred = base_dt.predict(X_test) # Use the model created to make predictions on the test data

# 	# (c) Top-DT: a better performing Decision Tree
# 	parameters = {'criterion':('gini', 'entropy'), 'max_depth':[3, 10], 'min_samples_split':[3, 5, 6]}
# 	top_dt = GridSearchCV(base_dt, parameters).fit(X_train, y_train)
# 	top_dt_y_test_pred = top_dt.predict(X_test) # Use the model created to make predictions on the test data

# 	# (d) PER: a Perceptron, with default parameter values
# 	perceptron = Perceptron().fit(X_train, y_train)
# 	perceptron_y_test_pred = perceptron.predict(X_test) # Use the model created to make predictions on the test data

# 	# (e) Base-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
# 	base_mlp = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd', max_iter=5000).fit(X_train, y_train)
# 	base_mlp_y_test_pred = base_mlp.predict(X_test) # Use the model created to make predictions on the test data

# 	# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search
# 	parameters = {'activation':('logistic', 'tanh', 'relu', 'identity'), 'hidden_layer_sizes':[(30, 50), (10, 10, 10)], 'solver':('adam', 'sgd')}
# 	top_mlp = GridSearchCV(base_mlp, parameters).fit(X_train, y_train)
# 	top_mlp_y_test_pred = top_mlp.predict(X_test) # Use the model created to make predictions on the test data

	# with open('drugs-performance.txt', 'a') as f:
	# 	f.writelines("\n----RUN #"+str(i)+"----\n\n")
	# 	a = "\n\n************************\n\nNB: a Gaussian Naive Bayes Classifier with the default parameters"
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, nb_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, nb_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, nb_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	# 	a = "\n\n************************\n\nBase-DT: a Decision Tree with the default parameters"
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, base_dt_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_dt_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_dt_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	# 	a = "\n\n************************\n\nTop-DT: a better performing Decision Tree"
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, top_dt_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_dt_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_dt_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	# 	a = "\n\n************************\n\nPER: a Perceptron, with default parameter values"
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, perceptron_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, perceptron_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, perceptron_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	# 	a = "\n\n************************\n\nBase-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters."
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, base_mlp_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_mlp_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, base_mlp_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

	# 	a = "\n\n************************\n\nTop-MLP: a better performing Multi-Layered Perceptron found using grid search"
	# 	f.writelines(a)
	# 	# The accuracy, macro-average F1 and weighted-average F1 of the model
	# 	accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, top_mlp_y_test_pred))
	# 	f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_mlp_y_test_pred, average='macro'))
	# 	f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, top_mlp_y_test_pred, average='weighted'))
	# 	f.writelines("\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

# 	'''
# 	NB: a Gaussian Naive Bayes Classifier with the default parameters
# 	'''
# 	accuracy = accuracy_score(y_test, nb_y_test_pred)
# 	f1_macro = f1_score(y_test, nb_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, nb_y_test_pred, average='weighted')

# 	avg_accuracy[0] += accuracy
# 	avg_macro_f1[0] += f1_macro
# 	avg_weighted_f1[0] += f1_weighted

# 	'''
# 	Base-DT: a Decision Tree with the default parameters
# 	'''
# 	accuracy = accuracy_score(y_test, base_dt_y_test_pred)
# 	f1_macro = f1_score(y_test, base_dt_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, base_dt_y_test_pred, average='weighted')
	
# 	avg_accuracy[1] += accuracy
# 	avg_macro_f1[1] += f1_macro
# 	avg_weighted_f1[1] += f1_weighted

# 	'''
# 	Top-DT: a better performing Decision Tree
# 	'''
# 	accuracy = accuracy_score(y_test, top_dt_y_test_pred)
# 	f1_macro = f1_score(y_test, top_dt_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, top_dt_y_test_pred, average='weighted')
	
# 	avg_accuracy[2] += accuracy
# 	avg_macro_f1[2] += f1_macro
# 	avg_weighted_f1[2] += f1_weighted

# 	'''
# 	PER: a Perceptron, with default parameter values
# 	'''
# 	accuracy = accuracy_score(y_test, perceptron_y_test_pred)
# 	f1_macro = f1_score(y_test, perceptron_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, perceptron_y_test_pred, average='weighted')
	
# 	avg_accuracy[3] += accuracy
# 	avg_macro_f1[3] += f1_macro
# 	avg_weighted_f1[3] += f1_weighted

# 	'''
# 	Base-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.
# 	'''
# 	accuracy = accuracy_score(y_test, base_mlp_y_test_pred)
# 	f1_macro = f1_score(y_test, base_mlp_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, base_mlp_y_test_pred, average='weighted')
	
# 	avg_accuracy[4] += accuracy
# 	avg_macro_f1[4] += f1_macro
# 	avg_weighted_f1[4] += f1_weighted

# 	'''
# 	Top-MLP: a better performing Multi-Layered Perceptron found using grid search
# 	'''
# 	accuracy = accuracy_score(y_test, top_mlp_y_test_pred)
# 	f1_macro = f1_score(y_test, top_mlp_y_test_pred, average='macro')
# 	f1_weighted = f1_score(y_test, top_mlp_y_test_pred, average='weighted')
	
# 	avg_accuracy[5] += accuracy
# 	avg_macro_f1[5] += f1_macro
# 	avg_weighted_f1[5] += f1_weighted

with open('drugs-performance.txt', 'a') as f:
	'''
	Average accuracy, average macro-average F1, average weighted-average F1 as well as the standard deviation for the accuracy,
	the standard deviation of the macro-average F1, and the standard deviation of the weighted-average F1 
	'''
	# f.writelines("\n\n----AVERAGES----\n")

	f.writelines("\n\n'''\nQ8 -\n'''\n")

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(nb_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(nb_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(nb_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((nb_accuracies-(np.sum(nb_accuracies)/10))**2/(len(nb_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((nb_macro_f1s-(np.sum(nb_macro_f1s)/10))**2/(len(nb_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((nb_weighted_f1s-(np.sum(nb_weighted_f1s)/10))**2/(len(nb_weighted_f1s)-1)))**.5)
	f.writelines("\n\nNB: a Gaussian Naive Bayes Classifier with the default parameters\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(base_dt_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(base_dt_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(base_dt_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((base_dt_accuracies-(np.sum(base_dt_accuracies)/10))**2/(len(base_dt_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((base_dt_macro_f1s-(np.sum(base_dt_macro_f1s)/10))**2/(len(base_dt_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((base_dt_weighted_f1s-(np.sum(base_dt_weighted_f1s)/10))**2/(len(base_dt_weighted_f1s)-1)))**.5)
	f.writelines("\n\nBase-DT: a Decision Tree with the default parameters\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(top_dt_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(top_dt_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(top_dt_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((top_dt_accuracies-(np.sum(top_dt_accuracies)/10))**2/(len(top_dt_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((top_dt_macro_f1s-(np.sum(top_dt_macro_f1s)/10))**2/(len(top_dt_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((top_dt_weighted_f1s-(np.sum(top_dt_weighted_f1s)/10))**2/(len(top_dt_weighted_f1s)-1)))**.5)
	f.writelines("\n\nTop-DT: a better performing Decision Tree\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(perceptron_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(perceptron_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(perceptron_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((perceptron_accuracies-(np.sum(perceptron_accuracies)/10))**2/(len(perceptron_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((perceptron_macro_f1s-(np.sum(perceptron_macro_f1s)/10))**2/(len(perceptron_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((perceptron_weighted_f1s-(np.sum(perceptron_weighted_f1s)/10))**2/(len(perceptron_weighted_f1s)-1)))**.5)
	f.writelines("\n\nPER: a Perceptron, with default parameter values\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(base_mlp_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(base_mlp_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(base_mlp_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((base_mlp_accuracies-(np.sum(base_mlp_accuracies)/10))**2/(len(base_mlp_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((base_mlp_macro_f1s-(np.sum(base_mlp_macro_f1s)/10))**2/(len(base_mlp_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((base_mlp_weighted_f1s-(np.sum(base_mlp_weighted_f1s)/10))**2/(len(base_mlp_weighted_f1s)-1)))**.5)
	f.writelines("\n\nBase-MLP: a Multi-Layered Perceptron with 1 hidden layer of 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters.\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)

	avg_accuracy = "Average accuracy score: " + "{:.2f}".format(np.sum(top_mlp_accuracies)/10)
	avg_macro_f1 = "Average macro-average F1 score: " + "{:.2f}".format(np.sum(top_mlp_macro_f1s)/10)
	avg_weighted_f1 = "Average weighted-average F1 score: " + "{:.2f}".format(np.sum(top_mlp_weighted_f1s)/10)
	accuracy_sd = "Accuracy standard deviation: " + "{:.2f}".format((np.sum((top_mlp_accuracies-(np.sum(top_mlp_accuracies)/10))**2/(len(top_mlp_accuracies)-1)))**.5)
	macro_f1_sd = "Macro F1 standard deviation: " + "{:.2f}".format((np.sum((top_mlp_macro_f1s-(np.sum(top_mlp_macro_f1s)/10))**2/(len(top_mlp_macro_f1s)-1)))**.5)
	weighted_f1_sd = "Weighted F1 standard deviation: " + "{:.2f}".format((np.sum((top_mlp_weighted_f1s-(np.sum(top_mlp_weighted_f1s)/10))**2/(len(top_mlp_weighted_f1s)-1)))**.5)
	f.writelines("\n\nTop-MLP: a better performing Multi-Layered Perceptron found using grid search\n\n" + avg_accuracy + "\n" + avg_macro_f1 + "\n" + avg_weighted_f1 + "\n" + accuracy_sd + "\n" + macro_f1_sd + "\n" + weighted_f1_sd)