# Women-cloth-review-prediction
Women Clothing Review Import Library

import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns

Import DataSet

df=pd.read_csv('/content/Womens Clothing E-Commerce Reviews.csv')

df.head()

df.info()

<class 'pandas.core.frame.DataFrame'> RangeIndex: 23486 entries, 0 to 23485 Data columns (total 10 columns):

Column Non-Null Count Dtype
0 Clothing ID 23486 non-null int64 1 Age 23486 non-null int64 2 Title 19676 non-null object 3 Review 22641 non-null object 4 Rating 23486 non-null int64 5 Recommended IND 23486 non-null int64 6 Positive Feedback Count 23486 non-null int64 7 Division Name 23472 non-null object 8 Department Name 23472 non-null object 9 Class Name 23472 non-null object dtypes: int64(5), object(5) memory usage: 1.8+ MB

df.shape

(23486, 10) Missing Values Remove missing values in Reviews Columns with no Review Text

df.isna().sum()

Clothing ID 0 Age 0 Title 3810 Review 845 Rating 0 Recommended IND 0 Positive Feedback Count 0 Division Name 14 Department Name 14 Class Name 14 dtype: int64

df[df['Review']==""]=np.NaN

df['Review'].fillna("No Review",inplace=True)

df.isna().sum()

Clothing ID 0 Age 0 Title 3810 Review 0 Rating 0 Recommended IND 0 Positive Feedback Count 0 Division Name 14 Department Name 14 Class Name 14 dtype: int64

df['Review']

0 Absolutely wonderful - silky and sexy and comfortable 1 Love this dress! it's sooo pretty. i happened to find it in a store, and i'm glad i did bc i never would have ordered it online bc it's petite. i bought a petite and am 5'8". i love the length on me- hits just a little below the knee. would definitely be a true midi on someone who is truly petite. 2 I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - ... 3 I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments! 4 This shirt is very flattering to all due to the adjustable front tie. it is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. love this shirt!!! ...
23481 I was very happy to snag this dress at such a great price! it's very easy to slip on and has a very flattering cut and color combo. 23482 It reminds me of maternity clothes. soft, stretchy, shiny material. cut is flattering and drapes nicely. i only found one button to close front... looked awkward. nice long sleeves.\nnot for me but maybe for others. just ok. 23483 This fit well, but the top was very see through. this never would have worked for me. i'm glad i was able to try it on in the store and didn't order it online. with different fabric, it would have been great. 23484 I bought this dress for a wedding i have this summer, and it's so cute. unfortunately the fit isn't perfect. the medium fits my waist perfectly, but was way too long and too big in the bust and shoulders. if i wanted to spend the money, i could get it tailored, but i just felt like it might not be worth it. side note - this dress was delivered to me with a nordstrom tag on it and i found it much cheaper there after looking! 23485 This dress in a lovely platinum is feminine and fits perfectly, easy to wear and comfy, too! highly recommend! Name: Review, Length: 23486, dtype: object Define Target (y) and Feature (X)

df.columns

Index(['Clothing ID', 'Age', 'Title', 'Review', 'Rating', 'Recommended IND', 'Positive Feedback Count', 'Division Name', 'Department Name', 'Class Name'], dtype='object')

X=df['Review']

y=df['Rating']

df['Rating'].value_counts()

5.0 13131 4.0 5077 3.0 2871 2.0 1565 1.0 842 Name: Rating, dtype: int64 Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, stratify = y, random_state=42529)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

((16440,), (7046,), (16440,), (7046,)) Get Feature Text Coversion to Tokens

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(lowercase=True, analyzer='word',ngram_range=(2,3),stop_words= 'english',max_features=5000)

X_train = cv.fit_transform(X_train)

cv.get_feature_names_out()

X_train.toarray()

array([[0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], ..., [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0]])

X_test=cv.fit_transform(X_test)

cv.get_feature_names_out()

array(['10 12', '10 dress', '10 dresses', ..., 'years old', 'yellow color', 'yoga pants'], dtype=object)

X_test.toarray()

array([[0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], ..., [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0], [0, 0, 0, ..., 0, 0, 0]]) Get Model Train

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)

MultinomialNB() Get Model Prediction

y_pred = model.predict(X_test)

y.pred.shape

y_pred

array([4., 5., 4., ..., 3., 4., 1.]) Get Probability of Each Predicted Class

model.predict_proba(X_test)

array([[0.01597287, 0.06247668, 0.02343906, 0.86805191, 0.03005948], [0.01938616, 0.03618749, 0.01366139, 0.02494917, 0.9058158 ], [0.12714164, 0.31224624, 0.02402702, 0.4310927 , 0.10549239], ..., [0.12935707, 0.07677888, 0.54537581, 0.10858167, 0.13990657], [0.01039322, 0.00244866, 0.02592397, 0.69603597, 0.26519819], [0.3715833 , 0.28173991, 0.19139055, 0.14724676, 0.00803948]]) Get Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))

[[ 17 24 29 47 136] [ 46 60 52 92 220] [ 91 87 115 184 384] [ 156 153 193 334 687] [ 395 302 375 737 2130]]

print(classification_report(y_test,y_pred))

          precision    recall  f1-score   support

     1.0       0.02      0.07      0.04       253
     2.0       0.10      0.13      0.11       470
     3.0       0.15      0.13      0.14       861
     4.0       0.24      0.22      0.23      1523
     5.0       0.60      0.54      0.57      3939

accuracy                           0.38      7046
macro avg 0.22 0.22 0.22 7046 weighted avg 0.41 0.38 0.39 7046

Recategories Ratings as poor(0) and good(1)

df['Rating'].value_counts()

5.0 13131 4.0 5077 3.0 2871 2.0 1565 1.0 842 Name: Rating, dtype: int64

df.replace({'Rating':{1:0,2:0,3:0,4:1,5:1}}, inplace=True)

y=df['Rating']

X=df['Review']

Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.7, stratify = y, random_state=42529)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

((16440,), (7046,), (16440,), (7046,)) Get Feature Text Conversion to Tokens

from sklearn.feature_extraction.text import CountVectorizer cv=CountVectorizer(lowercase=True, analyzer='word',ngram_range=(2, 3),stop_words= 'english', max_features=5000)

X_train = cv.fit_transform(X_train) X_test=cv.fit_transform(X_test)

Get Model Re-Train

from sklearn.naive_bayes import MultinomialNB model = MultinomialNB() model.fit(X_train, y_train)

MultinomialNB() Get Model Prediction

y_pred = model.predict(X_test) y_pred.shape

(7046,)

y_pred

array([1., 1., 1., ..., 1., 1., 0.]) Get Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))

[[ 443 1140] [1202 4261]]

print(classification_report(y_test,y_pred))

          precision    recall  f1-score   support

     0.0       0.27      0.28      0.27      1583
     1.0       0.79      0.78      0.78      5463

accuracy                           0.67      7046
macro avg 0.53 0.53 0.53 7046 weighted avg 0.67 0.67 0.67 7046
