
import os
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Q2 - Plot the distribution of the instances in each class and save the graphic in a file called BBC-distribution.pdf
classes = np.array(["business", "entertainment", "politics", "sport", "tech"])
class_article_count = np.zeros(classes.shape[0])

for i in range(len(classes)):
    direc = "BBC/" + classes[i]
    class_article_count[i] = len(os.listdir(direc))

plt.bar(classes,class_article_count)
plt.title('BBC Distribution')
plt.xlabel('Class Name')
plt.ylabel('Number of Instances')
plt.savefig("BBC-distribution.pdf")

# Q3 - Load the corpus using load files, this assigns a class to each article
files_info = sklearn.datasets.load_files(container_path='BBC/', encoding='latin1')

''' 
Q4 - Pre-process the dataset to have the features ready to be used by a multinomial Naive Bayes classifier.
This means that the frequency of each word in each class must be computed and stored in a term-document matrix.
'''
corpus = files_info.data # articles from all classes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) # Learns the vocabulary dictionary and return document-term matrix. [article#, word#, word_count]
y = files_info.target

'''
Q5 - Split the dataset into 80% for training and 20% for testing
X.toarray() transformed it into matrix where rows = articles and columns = words so, [row, column] = word count of that word[column] in that article[row]
'''
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=.2, random_state=None)

'''
Q6 - Train a multinomial Naive Bayes Classifier on the training set using the default parameters and evaluate it on the test set
'''
clf = MultinomialNB().fit(X_train, y_train) # Create a model using training data
y_test_pred = clf.predict(X_test) # Use the model created to make predictions on the test data

# Q7
with open('bbc-performance.txt', 'a') as f:
    # (a)
    a = "\n\n************************\n\n(a)\nMulti-nomialNB default values, try 1"
    f.writelines(a)

    # (b) The confusion matrix
    b = confusion_matrix(y_test, y_test_pred)
    f.writelines("\n\n(b)\n"+str(b))

    # (c) The precision, recall, and F1-measure for each class
    c = classification_report(y_test, y_test_pred, target_names=files_info.target_names)
    f.writelines("\n\n(c)\n"+str(c))

    # (d) The accuracy, macro-average F1 and weighted-average F1 of the model
    accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, y_test_pred))
    f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='macro'))
    f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='weighted'))
    f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

    # (e) The prior probability of each class
    prior_prob = class_article_count/np.sum(class_article_count)
    f.writelines("\n\n(e)\n")
    for i in range(prior_prob.shape[0]):
        f.writelines(classes[i] + ": " + "{:.2f}".format(prior_prob[i]) + "\n")

    # (f) The size of the vocabulary (i.e. the number of different words)
    voc_size = vectorizer.get_feature_names_out().shape[0] # Same as X.toarray().shape[1]! 29421 unique 'words'
    f.writelines("\n(f)\nThe size of the vocabulary is " + str(voc_size))

    # (g) The number of word-tokens in each class (i.e. the number of words in total2)
    class_word_count = np.zeros(classes.shape[0])
    f.writelines("\n\n(g)\n")
    for i in range(classes.shape[0]):
        class_word_count[i] = np.sum(clf.feature_count_[i,:]) # clf.deature_count_ returns a matrix where rows = classes and columns = features/vocabulary/words, so [row, column] = count of that word[column] in that class[row]
        f.writelines("The number of word-tokens in class " + classes[i] + " (" + str(i) + ") is " + str(class_word_count[i]) + "\n")

    # (h) The number of word-tokens in the entire corpus
    corpus_word_count = np.sum(class_word_count)
    f.writelines("\n(h)\nThe number of word-tokens in the entire corpus is " + str(corpus_word_count))

    # (i) The number and percentage of words with a frequency of zero in each class
    class_zero_freq_count, class_zero_freq_percentage = np.zeros(classes.shape[0]), np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        class_zero_freq_count[i] = np.count_nonzero(clf.feature_count_[i,:] == 0)

    class_zero_freq_percentage = class_zero_freq_count/voc_size # class_zero_freq_count/class_word_count
    f.writelines("\n\n(i)\n")
    for i in range(classes.shape[0]):
        f.writelines("The number and percentage of words with a frequency of zero in class " + classes[i] + " (" + str(i) + ") is " + str(int(class_zero_freq_count[i])) + " " + str(int(class_zero_freq_percentage[i]*100)) + "%\n")

    # (j) The number and percentage of words with a frequency of zero one in the entire corpus
    corpus_one_freq_count = np.count_nonzero(clf.feature_count_ == 1)
    corpus_one_freq_percentage = corpus_one_freq_count/voc_size # corpus_one_freq_count/corpus_word_count
    f.writelines("\n(j)\nThe number and percentage of words with a frequency of one in the entire corpus is " + str(int(corpus_one_freq_count)) + " " + str(int(corpus_one_freq_percentage*100)) + "%")

    # (k) 2 favorite words (that are present in the vocabulary) and their log-prob
    pizza_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['pizza']] # clf.feature_log_prob_ is a matrix where rows = classes and columns = features/vocabulary/words
    coffee_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['coffee']] # vectorizer.vocabulary_ returns the index of the word
    f.writelines("\n\n(k)\nThe log-prob of the word 'coffee' in each class of the 5 classes is: " + str(coffee_log_prob))
    f.writelines("\nThe log-prob of the word 'pizza' in each class of the 5 classes is: " + str(pizza_log_prob))

'''
Q8 - Redo steps 6 and 7 without changing anything
'''
clf = MultinomialNB().fit(X_train, y_train) # Create a model using training data
y_test_pred = clf.predict(X_test) # Use the model created to make predictions on the test data

with open('bbc-performance.txt', 'a') as f:
    # (a)
    a = "\n\n************************\n\n(a)\nMulti-nomialNB default values, try 2"
    f.writelines(a)

    # (b) The confusion matrix
    b = confusion_matrix(y_test, y_test_pred)
    f.writelines("\n\n(b)\n"+str(b))

    # (c) The precision, recall, and F1-measure for each class
    c = classification_report(y_test, y_test_pred, target_names=files_info.target_names)
    f.writelines("\n\n(c)\n"+str(c))

    # (d) The accuracy, macro-average F1 and weighted-average F1 of the model
    accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, y_test_pred))
    f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='macro'))
    f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='weighted'))
    f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

    # (e) The prior probability of each class
    prior_prob = class_article_count/np.sum(class_article_count)
    f.writelines("\n\n(e)\n")
    for i in range(prior_prob.shape[0]):
        f.writelines(classes[i] + ": " + "{:.2f}".format(prior_prob[i]) + "\n")

    # (f) The size of the vocabulary (i.e. the number of different words)
    voc_size = vectorizer.get_feature_names_out().shape[0] # Same as X.toarray().shape[1]! 29421 unique 'words'
    f.writelines("\n(f)\nThe size of the vocabulary is " + str(voc_size))

    # (g) The number of word-tokens in each class (i.e. the number of words in total2)
    class_word_count = np.zeros(classes.shape[0])
    f.writelines("\n\n(g)\n")
    for i in range(classes.shape[0]):
        class_word_count[i] = np.sum(clf.feature_count_[i,:]) # clf.deature_count_ returns a matrix where rows = classes and columns = features/vocabulary/words, so [row, column] = count of that word[column] in that class[row]
        f.writelines("The number of word-tokens in class " + classes[i] + " (" + str(i) + ") is " + str(class_word_count[i]) + "\n")

    # (h) The number of word-tokens in the entire corpus
    corpus_word_count = np.sum(class_word_count)
    f.writelines("\n(h)\nThe number of word-tokens in the entire corpus is " + str(corpus_word_count))

    # (i) The number and percentage of words with a frequency of zero in each class
    class_zero_freq_count, class_zero_freq_percentage = np.zeros(classes.shape[0]), np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        class_zero_freq_count[i] = np.count_nonzero(clf.feature_count_[i,:] == 0)

    class_zero_freq_percentage = class_zero_freq_count/voc_size # class_zero_freq_count/class_word_count
    f.writelines("\n\n(i)\n")
    for i in range(classes.shape[0]):
        f.writelines("The number and percentage of words with a frequency of zero in class " + classes[i] + " (" + str(i) + ") is " + str(int(class_zero_freq_count[i])) + " " + str(int(class_zero_freq_percentage[i]*100)) + "%\n")

    # (j) The number and percentage of words with a frequency of zero one in the entire corpus
    corpus_one_freq_count = np.count_nonzero(clf.feature_count_ == 1)
    corpus_one_freq_percentage = corpus_one_freq_count/voc_size # corpus_one_freq_count/corpus_word_count
    f.writelines("\n(j)\nThe number and percentage of words with a frequency of one in the entire corpus is " + str(int(corpus_one_freq_count)) + " " + str(int(corpus_one_freq_percentage*100)) + "%")

    # (k) 2 favorite words (that are present in the vocabulary) and their log-prob
    pizza_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['pizza']] # clf.feature_log_prob_ is a matrix where rows = classes and columns = features/vocabulary/words
    coffee_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['coffee']] # vectorizer.vocabulary_ returns the index of the word
    f.writelines("\n\n(k)\nThe log-prob of the word 'coffee' in each class of the 5 classes is: " + str(coffee_log_prob))
    f.writelines("\nThe log-prob of the word 'pizza' in each class of the 5 classes is: " + str(pizza_log_prob))

'''
Q9 - Redo steps 6 and 7 again, but this time, change the smoothing value to 0.0001
'''
clf = MultinomialNB(alpha=.0001).fit(X_train, y_train) # Create a model using training data
y_test_pred = clf.predict(X_test) # Use the model created to make predictions on the test data

with open('bbc-performance.txt', 'a') as f:
    # (a)
    a = "\n\n************************\n\n(a)\nMulti-nomialNB with 0.0001 smoothing"
    f.writelines(a)

    # (b) The confusion matrix
    b = confusion_matrix(y_test, y_test_pred)
    f.writelines("\n\n(b)\n"+str(b))

    # (c) The precision, recall, and F1-measure for each class
    c = classification_report(y_test, y_test_pred, target_names=files_info.target_names)
    f.writelines("\n\n(c)\n"+str(c))

    # (d) The accuracy, macro-average F1 and weighted-average F1 of the model
    accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, y_test_pred))
    f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='macro'))
    f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='weighted'))
    f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

    # (e) The prior probability of each class
    prior_prob = class_article_count/np.sum(class_article_count)
    f.writelines("\n\n(e)\n")
    for i in range(prior_prob.shape[0]):
        f.writelines(classes[i] + ": " + "{:.2f}".format(prior_prob[i]) + "\n")

    # (f) The size of the vocabulary (i.e. the number of different words)
    voc_size = vectorizer.get_feature_names_out().shape[0] # Same as X.toarray().shape[1]! 29421 unique 'words'
    f.writelines("\n(f)\nThe size of the vocabulary is " + str(voc_size))

    # (g) The number of word-tokens in each class (i.e. the number of words in total2)
    class_word_count = np.zeros(classes.shape[0])
    f.writelines("\n\n(g)\n")
    for i in range(classes.shape[0]):
        class_word_count[i] = np.sum(clf.feature_count_[i,:]) # clf.deature_count_ returns a matrix where rows = classes and columns = features/vocabulary/words, so [row, column] = count of that word[column] in that class[row]
        f.writelines("The number of word-tokens in class " + classes[i] + " (" + str(i) + ") is " + str(class_word_count[i]) + "\n")

    # (h) The number of word-tokens in the entire corpus
    corpus_word_count = np.sum(class_word_count)
    f.writelines("\n(h)\nThe number of word-tokens in the entire corpus is " + str(corpus_word_count))

    # (i) The number and percentage of words with a frequency of zero in each class
    class_zero_freq_count, class_zero_freq_percentage = np.zeros(classes.shape[0]), np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        class_zero_freq_count[i] = np.count_nonzero(clf.feature_count_[i,:] == 0)

    class_zero_freq_percentage = class_zero_freq_count/voc_size # class_zero_freq_count/class_word_count
    f.writelines("\n\n(i)\n")
    for i in range(classes.shape[0]):
        f.writelines("The number and percentage of words with a frequency of zero in class " + classes[i] + " (" + str(i) + ") is " + str(int(class_zero_freq_count[i])) + " " + str(int(class_zero_freq_percentage[i]*100)) + "%\n")

    # (j) The number and percentage of words with a frequency of zero one in the entire corpus
    corpus_one_freq_count = np.count_nonzero(clf.feature_count_ == 1)
    corpus_one_freq_percentage = corpus_one_freq_count/voc_size # corpus_one_freq_count/corpus_word_count
    f.writelines("\n(j)\nThe number and percentage of words with a frequency of one in the entire corpus is " + str(int(corpus_one_freq_count)) + " " + str(int(corpus_one_freq_percentage*100)) + "%")

    # (k) 2 favorite words (that are present in the vocabulary) and their log-prob
    pizza_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['pizza']] # clf.feature_log_prob_ is a matrix where rows = classes and columns = features/vocabulary/words
    coffee_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['coffee']] # vectorizer.vocabulary_ returns the index of the word
    f.writelines("\n\n(k)\nThe log-prob of the word 'coffee' in each class of the 5 classes is: " + str(coffee_log_prob))
    f.writelines("\nThe log-prob of the word 'pizza' in each class of the 5 classes is: " + str(pizza_log_prob))

'''
Q10 - Redo steps 6 and 7 again, but this time, change the smoothing value to 0.9
'''
clf = MultinomialNB(alpha=.9).fit(X_train, y_train) # Create a model using training data
y_test_pred = clf.predict(X_test) # Use the model created to make predictions on the test data

with open('bbc-performance.txt', 'a') as f:
    # (a)
    a = "\n\n************************\n\n(a)\nMulti-nomialNB with 0.9 smoothing"
    f.writelines(a)

    # (b) The confusion matrix
    b = confusion_matrix(y_test, y_test_pred)
    f.writelines("\n\n(b)\n"+str(b))

    # (c) The precision, recall, and F1-measure for each class
    c = classification_report(y_test, y_test_pred, target_names=files_info.target_names)
    f.writelines("\n\n(c)\n"+str(c))

    # (d) The accuracy, macro-average F1 and weighted-average F1 of the model
    accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, y_test_pred))
    f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='macro'))
    f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='weighted'))
    f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

    # (e) The prior probability of each class
    prior_prob = class_article_count/np.sum(class_article_count)
    f.writelines("\n\n(e)\n")
    for i in range(prior_prob.shape[0]):
        f.writelines(classes[i] + ": " + "{:.2f}".format(prior_prob[i]) + "\n")

    # (f) The size of the vocabulary (i.e. the number of different words)
    voc_size = vectorizer.get_feature_names_out().shape[0] # Same as X.toarray().shape[1]! 29421 unique 'words'
    f.writelines("\n(f)\nThe size of the vocabulary is " + str(voc_size))

    # (g) The number of word-tokens in each class (i.e. the number of words in total2)
    class_word_count = np.zeros(classes.shape[0])
    f.writelines("\n\n(g)\n")
    for i in range(classes.shape[0]):
        class_word_count[i] = np.sum(clf.feature_count_[i,:]) # clf.deature_count_ returns a matrix where rows = classes and columns = features/vocabulary/words, so [row, column] = count of that word[column] in that class[row]
        f.writelines("The number of word-tokens in class " + classes[i] + " (" + str(i) + ") is " + str(class_word_count[i]) + "\n")

    # (h) The number of word-tokens in the entire corpus
    corpus_word_count = np.sum(class_word_count)
    f.writelines("\n(h)\nThe number of word-tokens in the entire corpus is " + str(corpus_word_count))

    # (i) The number and percentage of words with a frequency of zero in each class
    class_zero_freq_count, class_zero_freq_percentage = np.zeros(classes.shape[0]), np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        class_zero_freq_count[i] = np.count_nonzero(clf.feature_count_[i,:] == 0)

    class_zero_freq_percentage = class_zero_freq_count/voc_size # class_zero_freq_count/class_word_count
    f.writelines("\n\n(i)\n")
    for i in range(classes.shape[0]):
        f.writelines("The number and percentage of words with a frequency of zero in class " + classes[i] + " (" + str(i) + ") is " + str(int(class_zero_freq_count[i])) + " " + str(int(class_zero_freq_percentage[i]*100)) + "%\n")

    # (j) The number and percentage of words with a frequency of zero one in the entire corpus
    corpus_one_freq_count = np.count_nonzero(clf.feature_count_ == 1)
    corpus_one_freq_percentage = corpus_one_freq_count/voc_size # corpus_one_freq_count/corpus_word_count
    f.writelines("\n(j)\nThe number and percentage of words with a frequency of one in the entire corpus is " + str(int(corpus_one_freq_count)) + " " + str(int(corpus_one_freq_percentage*100)) + "%")

    # (k) 2 favorite words (that are present in the vocabulary) and their log-prob
    pizza_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['pizza']] # clf.feature_log_prob_ is a matrix where rows = classes and columns = features/vocabulary/words
    coffee_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['coffee']] # vectorizer.vocabulary_ returns the index of the word
    f.writelines("\n\n(k)\nThe log-prob of the word 'coffee' in each class of the 5 classes is: " + str(coffee_log_prob))
    f.writelines("\nThe log-prob of the word 'pizza' in each class of the 5 classes is: " + str(pizza_log_prob))
