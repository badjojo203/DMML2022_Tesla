# Exemple flow
<p>Here is an exemple of implémentation that use ourHelper lib</p>
please go to <a href="documentation.md">documentation</a> for more information on the library used 

#### init the lib
    from lib.DataVectorizer import DataVectorizer
    from lib.DataManager import DataManager
    from lib.DataCleaner import DataCleaner
    from lib.ClassifierManager import ClassifierManager

### instance the objects
    classifiermanager = ClassifierManager()
    datavectorizer = DataVectorizer()
    datacleaner = DataCleaner()

### load the data
we set the difficulty column as the target column
and the sentences column as the data column

    df = pd.read_csv('../data/training_data.csv')
    datamanager = DataManager(df,"difficulty","sentence")
    datamanager.labelEncode("difficulty")

### split the data
    datamanager.splitData(0.2,0)

### clean the data
    datacleaner.clean(datamanager,["lemmatize"])

### vectorize the data
    datavectorizer.fit("tfidf",datamanager)

### train the model
    classifiermanager.fit_predict("logistic_regression",datamanager,verbose=True)

### export csv
    datamanager.inferDataToCSV(df_pred,log_reg,"tfidf","sentence","sentence","result")

# Searching best parameters
the provides api that allow you to search the 
best cleaner/vectorizer/classier combination but as well 
the best parameters for a classifier or the train
test split

### search best combination

    classifiermanager.searchBestCombinations(datamanager,["tfidf","count"],["linear_svc","knn","naive_bayes"],["empty"])

### search the pest test size parameter for split

    method = classifiermanager.getClassifier("linear_svc")
    vectorizer = datavectorizer.getVectorizer("tfidf")
    
    datamanager.otpimize_test_size(method,vectorizer,np.arange(0.0241,0.0242,0.00001),verbose=True)classifiermanager.searchBestTestSize(datamanager,["tfidf","count"],["linear_svc","knn","naive_bayes"],["empty"])

### search the best random state parameter for split

    method = classifiermanager.getClassifier("linear_svc")
    vectorizer = datavectorizer.getVectorizer("tfidf")
    
    datamanager.otpimize_randomstate(method,vectorizer,np.arange(0,10),verbose=True,test_size=0.024169999999999997)

### search the best parameters for a classifier

    param_grid = {
    'alpha': np.arange(0.001,0.01,0.001),
    "fit_prior":[True,False]
    }

    datavectorizer.fit("tfidf",datamanager)
    
    classifiermanager.gridSearch("naive_bayes",param_grid,datamanager)
