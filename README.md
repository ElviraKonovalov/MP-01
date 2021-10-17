# MP-01
This mini-project experiment with different machine learning algorithms and different data sets using python's scikit-learn Library.
The focus of this mini-project lies more on the experimentations and analysis than on the implementation.

## Installation
*Assuming `python3.7` and `venv` pacakage are installed.* <br>
1. Create a `python` virtual environment inside the `MP-01` folder <br>
```shell
cd MP-01
python3.7 -m venv myenv
```
2. Activate the environment <br>
```shell
source myenv/bin/activate
```
3. Install packages<br>
```shell
pip install pandas
pip install matplotlib
pip install -U scikit-learn
pip install numpy
```
4. Run script<br>
```shell
python3 Task#.py
```
*where Task# is Task1 or Task2*
## Dependencies
<b>Task1</b> requires the `BBC/` directory containing the BBC news article dataset.<br>
<b>task2</b> requires the CSV file `drugs200.csv` containing the drugs dataset.
## Imports
<b>Task 1</b> requires the following packages and libraries:
```python
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
```
<b>Task 2</b> requires the following packages and libraries:
```python
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
```
