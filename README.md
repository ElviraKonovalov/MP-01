# MP-01
This mini-project experiments with different machine learning algorithms and different datasets using <i>python's scikit-learn Library</i>.
The focus of this mini-project lies more on the experimentations and analysis than on the implementation.

## Installation
*Assuming `python3.9` and `venv` pacakage are installed.* <br>
1. Create a `python` virtual environment inside the `MP-01` folder <br>
```shell
cd MP-01
python3.9 -m venv myenv
```
2. Activate the environment <br>
```shell
source myenv/bin/activate
```
3. Install any non-built-in modules or packages<br>
```shell
pip install pandas
pip install matplotlib
pip install -U scikit-learn
pip install numpy
```
4. Run scripts<br>
```shell
# to run Task1
python3 Task1.py
```
```shell
# to run Task2
python3 Task2.py
```
##### Notes: 
1. `Task2.py` returns a warning message that can be ignored.
2. `Task2.py` takes a little longer than `Task1.py` to run.
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
## Output
<b>Task</b> 1 outputs:
1. `bbc-distribution.pdf`
2. `bbc-performance.txt`

<b>Task 2</b> outputs:
1. `drug-distribution.pdf`
2. `drug-performance.txt`

## Presentation and Discussion Files
The following files are for Demo presentation purposes:
1. `MP1-Presentation.ipynb`
2. `MP1-Presentation.pdf`<br>

The following are discussion text files:
1. `bbc-discussion.txt`
2. `drug-discussion.txt`
