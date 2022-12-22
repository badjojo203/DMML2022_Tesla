
*   [Data manager](#datamanager)
*   [Data cleaner](#datacleaner)
*   [Data vectorizer](#datavectorizer)
*   [Classifier manager](#classifiermanager)

Description
-----------

DMML HELPER library.

Classes
-------

### DataManager

The DataManager class is a utility class that provides methods for preprocessing data and evaluating the performance of classifiers. It can be used to split data into training and testing sets, label encode categorical data, and optimize the test size and random state parameters for a given classifier and vectorizer.

#### Example usage

`manager = DataManager(data, 'y_column_name', ['x1', 'x2', 'x3'])   manager.labelEncode('y_column_name')   manager.splitData()   manager.optimize_test_size(classifier, vectorizer, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])`

#### Attributes

*   **y\_column\_name** (str) - The name of the column containing the labels for the data.
*   **y** (pandas series) - The labels for the data.
*   **X** (pandas dataframe) - The features for the data.
*   **y\_test** (pandas series) - The labels for the test data.
*   **y\_train** (pandas series) - The labels for the training data.
*   **X\_test** (pandas dataframe) - The features for the test data.
*   **X\_train** (pandas dataframe) - The features for the training data.
*   **data** (pandas dataframe) - The original data.
*   **initial\_data** (pandas dataframe) - The original data, saved before any preprocessing is applied.
*   **initial\_y\_test** (pandas series) - The labels for the test data, saved before any preprocessing is applied.
*   **initial\_y\_train** (pandas series) - The labels for the training data, saved before any preprocessing is applied.
*   **initial\_X\_test** (pandas dataframe) - The features for the test data, saved before any preprocessing is applied.
*   **initial\_X\_train** (pandas dataframe) - The features for the training data, saved before any preprocessing is applied.
*   **encoder** (sklearn.preprocessing.LabelEncoder) - An object for label encoding categorical data.

#### Méthodes

##### \_\_init\_\_

The class constructor.

*   **data** (pandas dataframe) - The data to be used in the analysis.
*   **y\_column\_name** (string) - The name of the column in the data that contains the target variable.
*   **x\_column\_names** (list of strings) - A list of the names of the columns in the data that contain the features for the model.

##### labelEncode

Label encodes a specified column in the data.

*   **column\_name** (string) - The name of the column to be label encoded.

##### splitData

Splits the data into training and testing sets.

*   **test\_size** (float, optional) - The proportion of the data to be used for testing. Default is 0.2.
*   **random\_state** (int, optional) - The seed used by the random number generator. Default is 0.

##### otpimize\_test\_size

Finds the optimal test size for a given classifier and vectorizer by testing different test sizes and returning the one with the highest accuracy.

*   **method** (classifier object) - The classifier to be used.
*   **vectorizer** (vectorizer object) - The vectorizer to be used.
*   **test\_size\_array** (list of floats) - An array of test sizes to be tested.
*   **verbose** (bool, optional) - If True, prints the optimal test size and its corresponding accuracy score. Default is False.

Returns:

*   **best\_params** (float) - The optimal test size.
*   **best\_score** (float) - The corresponding accuracy score for the optimal test size.

##### otpimize\_randomstate

A method that optimizes the value of the random\_state parameter of the train\_test\_split function. It returns the optimal value of the random\_state parameter and the corresponding score.

*   **method** (object) - A classifier object that has a fit and predict method.
*   **vectorizer** (object) - A vectorizer object that has a fit\_transform and transform method.
*   **random\_state\_array** (list) - A list of random states values e.g. \[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\].
*   **verbose** (bool) - If True, print the best value of the random\_state parameter and the corresponding score.
*   **test\_size** (float) - The test size parameter of the train\_test\_split function.

Returns:

*   **best\_params** (int) - The optimal value of the random\_state parameter.
*   **best\_score** (float) - The corresponding score of the optimal value of the random\_state parameter.

##### inferDataToCSV

A method that infers the data and saves it to a CSV file.

*   **model** (Type) - The model that will be used to make the predictions.
*   **vectorizer** (Type) - The vectorizer that will be used to transform the data.
*   **filename** (str) - The name of the CSV file that will be created.

##### accuracy\_conf\_mat

A method that calculates the accuracy and confusion matrix for a given model and vectorizer.

*   **model** (Type) - The model that will be used to make the predictions.
*   **vectorizer** (Type) - The vectorizer that will be used to transform the data.
*   **X\_train** (Type) - The training data.
*   **y\_train** (Type) - The training labels.
*   **X\_test** (Type) - The test data.
*   **y\_test** (Type) - The test labels.
*   **target\_names** (Type) - The names of the target classes.

### DataCleaner

A class for cleaning and preprocessing data for text classification.

#### Exemple usage

`sample code`

#### Attributes

*   **cleaner** (dict) - A dictionary containing methods for cleaning data.

#### Méthodes

##### \_\_init\_\_

The class constructor.

##### remove\_stop\_words

A method for removing stop words from a sentence.

*   **sentence** (str) - The input sentence to remove stop words from.

##### lemmatize

A method for lemmatizing the words in a sentence.

*   **sentence** (str) - The input sentence to lemmatize.

##### data\_cleaner

A method for cleaning data.

*   **sms** (str) - The input data to clean.

##### empty

A method that does nothing.

*   **sentence** (str) - The input sentence to return unchanged.

##### clean

A method for applying cleaning methods to the data in a DataManager object.

*   **datamanager** (DataManager) - The DataManager object containing the data to clean.
*   **names** (list) - A list of strings representing the names of the cleaning methods to apply.

##### getCleaners

A method for getting a list of cleaning methods.

*   **names** (list) - A list of strings representing the names of the cleaning methods to return.

##### unClean

A method to reset the data to its original form by setting the X\_test and X\_train attributes to their original values stored in initial\_X\_test and initial\_X\_train, respectively.

*   **datamanager** (DataManager object) - An object of the DataManager class which holds the data to be reset.

### DataVectorizer

A class for vectorizing data for use in machine learning models.

#### Example usage

`vectorizer = DataVectorizer()   vectorizer.fit("count", datamanager)   model.fit(datamanager.X_train, datamanager.y_train)`

#### Attributs

*   **vectorizer** (dictionary) - A dictionary containing different vectorization objects as values, with keys for easy access.

#### Méthodes

##### \_\_init\_\_

The class constructor.

##### setVectorizer

A method for setting a vectorization object in the vectorizer dictionary.

*   **name** (string) - The key for accessing the vectorization object in the dictionary.
*   **vectorizer** (object) - The vectorization object to be added to the dictionary.

##### fit

This method takes the name of a vectorization method and a DataManager object as inputs and fits the vectorization method to the data in the DataManager object. It updates the DataManager object's X\_train and X\_test attributes with the transformed data. The method is called like this:

`DataVectorizer.fit(name, datamanager)`

*   **name** (str) - The name of the vectorization method to use. Can be 'hash', 'count', or 'tfidf'.
*   **datamanager** (DataManager) - The DataManager object containing the data to be transformed.

##### unvectorize

A method that sets the X\_test and X\_train attributes of the DataManager object back to their original form (before the vectorization).

*   **datamanager** (DataManager) - A DataManager object containing the X\_test and X\_train attributes to be modified.

##### getVectorizers

A method that returns a list of the specified vectorizers.

*   **names** (list of str) - A list of the names of the vectorizers to be returned.

### ClassifierManager

A class that manages different types of classifiers and provides functions for training, evaluating and predicting with them.

#### Example usage

`classifier_manager = ClassifierManager() classifier_manager.displayNames() classifier_manager.fit_predict('random_forest', datamanager, True)`

#### Attributes

*   **classifier** (dict) - A dictionary containing different classifiers as values, with their names as keys.

#### Methods

##### \_\_init\_\_

The class constructor. Initializes the classifier manager with a dictionary containing several classifier algorithms, including Multinomial Naive Bayes, Logistic Regression, Random Forest, Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Linear SVM, and Voting Classifier. The Voting Classifier is initialized with an empty list of estimators and voting set to "hard".

*   **self** (ClassifierManager) - The object instance of the class.

##### setClassifier

Sets the specified classifier in the classifier dictionary with the given name to the given classifier object.

*   **self** (ClassifierManager) - The object instance of the class.
*   **name** (str) - The name of the classifier in the dictionary.
*   **classifier** (object) - The classifier object to set in the dictionary.

##### displayNames

Prints the names of all classifiers in the classifier dictionary.

*   **self** (ClassifierManager) - The object instance of the class.

##### initVotingClassifier

Initializes the Voting Classifier in the classifier dictionary with the given list of classifiers as estimators. The voting method is set to "hard".

*   **self** (ClassifierManager) - The object instance of the class.
*   **classifiers\_names** (list) - A list of strings representing the names of the classifiers to use as estimators in the Voting Classifier.

##### getClassifiers

Returns a list of classifiers that match the given names.

*   **names** (list of strings) - A list of the names of the classifiers to return.

##### getClassifier

Returns a single classifier that matches the given name.

*   **name** (string) - The name of the classifier to return.

##### evaluate

Prints evaluation metrics for the given true labels and predicted labels.

*   **y\_test** (array-like) - The true labels of the test data.
*   **y\_pred** (array-like) - The predicted labels of the test data.

##### fit\_predict

A method that trains and tests a classifier on a given data set and returns the classifier. It takes as input the name of the classifier, a DataManager object containing the data, a boolean indicating whether or not to print evaluation metrics, and optional parameters to set on the classifier.

*   **name** (string) - The name of the classifier to use.
*   **datamanager** (DataManager) - The data to use for training and testing.
*   **verbose** (bool) - A flag indicating whether or not to print evaluation metrics.
*   **params** (dict) - Optional parameters to set on the classifier.

Returns: the trained classifier.

##### predict

This method takes a `datamanager` object and a `classifier` object as input and returns the predictions made by the classifier on the data stored in the `datamanager` object.

*   **datamanager** (DataManager) - An object of the `DataManager` class that holds the data to make predictions on.
*   **classifier** (sklearn classifier) - A scikit-learn classifier object used to make predictions.

##### fit\_predict\_all

This method takes a `datamanager` object and a boolean `verbose` flag as input and fits and predicts with all the classifiers stored in the `classifier` attribute of the `ClassifierManager` object on the data stored in the `datamanager` object. If `verbose` is set to `True`, the evaluation metrics for each classifier will be printed.

*   **datamanager** (DataManager) - An object of the `DataManager` class that holds the data to make predictions on.
*   **verbose** (bool) - A flag that indicates whether to print the evaluation metrics for each classifier or not.

##### searchBestCombinations

This method takes a `datamanager` object, three lists of strings `vectorizers_names`, `classifiers_names` and `data_cleaners_names` as input and searches for the best combination of vectorizers, classifiers and data cleaners for the data stored in the `datamanager` object using a grid search. It returns a list of dictionaries containing the results for each combination of methods.

*   **datamanager** (DataManager) - An object of the `DataManager` class that holds the data to make predictions on.
*   **vectorizers\_names** (list of str) - A list of strings representing the names of the vectorizers to use in the grid search.
*   **classifiers\_names** (list of str) - A list of strings representing the names of the classifiers to use in the grid search.
*   **data\_cleaners\_names** (list of str) - A list of strings representing the names of the data cleaners to use in the grid search.

##### gridSearch

This method performs a grid search to find the best combination of hyperparameters for a given classifier and data vectorizer. It takes in three arguments:

*   **classifier\_name** (str) - The name of the classifier to use for the grid search.
*   **vectorizer\_name** (str) - The name of the data vectorizer to use for the grid search.
*   **param\_grid** (dict) - A dictionary containing the hyperparameters to test and the values to test for each hyperparameter.

This method returns a tuple containing the best classifier and data vectorizer after the grid search has been completed.

#### Example usage

`best_classifier, best_vectorizer = classifier_manager.gridSearch("random_forest", "tfidf", {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]})`
