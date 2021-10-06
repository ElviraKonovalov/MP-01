
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

# Q2
classes = np.array(["business", "entertainment", "politics", "sport", "tech"])
class_article_count = np.zeros(classes.shape[0])

for i in range(len(classes)):
    direc = "BBC/" + classes[i]
    class_article_count[i] = len(os.listdir(direc))
    # class_counts.append(len(os.listdir(direc)))

plt.bar(classes,class_article_count)
plt.title('BBC Distribution')
plt.xlabel('Class Name')
plt.ylabel('Number of Instances')
plt.savefig("BBC-distribution.pdf")

# Q3
files_info = sklearn.datasets.load_files(container_path='BBC/', encoding='latin1')
# for i in range(len(files_info.data)):
#     print("Text:",files_info.data[i],"\n-----Class number:",files_info.target[i])

# Q4
corpus = files_info.data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = files_info.target

# Q5
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=.2, random_state=None)

# Q6
clf = MultinomialNB(alpha=0.9).fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

# Q7
with open('bbc-performance.txt', 'a') as f:
    # (a)
    a = "\n\n************************\n\n(a)\nMulti-nomialNB smoothing value 0.9"
    f.writelines(a)

    # (b)
    b = confusion_matrix(y_test, y_test_pred)
    f.writelines("\n\n(b)\n"+str(b))

    # (c)
    c = classification_report(y_test, y_test_pred, target_names=files_info.target_names)
    f.writelines("\n\n(c)\n"+str(c))

    # (d)
    accuracy = "Accuracy score: " + "{:.2f}".format(accuracy_score(y_test, y_test_pred))
    f1_macro = "Macro-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='macro'))
    f1_weighted = "Weighted-average F1 score: " + "{:.2f}".format(f1_score(y_test, y_test_pred, average='weighted'))
    f.writelines("\n\n(d)\n" + accuracy + "\n" + f1_macro + "\n" + f1_weighted)

    # (e)
    prior_prob = class_article_count/np.sum(class_article_count)
    f.writelines("\n\n(e)\n")
    for i in range(prior_prob.shape[0]):
        f.writelines(classes[i] + ": " + "{:.2f}".format(prior_prob[i]) + "\n")

    # (f)
    voc_size = vectorizer.get_feature_names_out().shape[0]
    # vectorizer.vocabulary_
    f.writelines("\n(f)\nThe size of the vocabulary is " + str(voc_size))

    # (g)
    # print(X.toarray().shape)
    # print(np.sum(class_counts))
    class_word_count = np.zeros(classes.shape[0])
    f.writelines("\n\n(g)\n")
    for i in range(classes.shape[0]):
        class_word_count[i] = np.sum(clf.feature_count_[i,:])
        f.writelines("The number of word-tokens in class " + classes[i] + " (" + str(i) + ") is " + str(class_word_count[i]) + "\n")

    # (h)
    corpus_word_count = np.sum(class_word_count)
    f.writelines("\n(h)\nThe number of word-tokens in the entire corpus is " + str(corpus_word_count))

    # (i)
    class_zero_freq_count, class_zero_freq_percentage = np.zeros(classes.shape[0]), np.zeros(classes.shape[0])
    for i in range(classes.shape[0]):
        class_zero_freq_count[i] = np.count_nonzero(clf.feature_count_[i,:] == 0)

    class_zero_freq_percentage = class_zero_freq_count/class_word_count
    f.writelines("\n\n(i)\n")
    for i in range(classes.shape[0]):
        f.writelines("The number and percentage of words with a frequency of zero in class " + classes[i] + " (" + str(i) + ") is " + str(int(class_zero_freq_count[i])) + " " + str(int(class_zero_freq_percentage[i]*100)) + "%\n")

    # (j)
    corpus_one_freq_count = np.count_nonzero(clf.feature_count_ == 1)
    corpus_one_freq_percentage = corpus_one_freq_count/corpus_word_count
    f.writelines("\n(j)\nThe number and percentage of words with a frequency of one in the entire corpus is " + str(int(corpus_one_freq_count)) + " " + str(int(corpus_one_freq_percentage*100)) + "%")

    # (k)
    pizza_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['pizza']]
    coffee_log_prob = clf.feature_log_prob_[:,vectorizer.vocabulary_['coffee']]
    f.writelines("\n\n(k)\nThe log-prob of the word 'coffee' in each class of the 5 classes is: " + str(coffee_log_prob))
    f.writelines("\nThe log-prob of the word 'pizza' in each class of the 5 classes is: " + str(pizza_log_prob))




