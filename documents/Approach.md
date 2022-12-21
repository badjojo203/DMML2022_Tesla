<h2>Data Cleaning</h2>
<p>It is a set of techniques for preparing data for use in analysis and processing. Data cleaning can include operations such as removing empty words and clean the data.</p>
<h2>Vectorization</h2>
<p>It is a data pre-processing step consisting in transforming textual data into numerical feature vectors. This can be useful for machine learning algorithms that cannot process textual data directly. There are several vectorization methods, such as hashing, TF-IDF and CountVectorizer.</p>
<h4>Hashing</h4>
<p>It is a vectorization method that transforms text into a vector of numerical features using a hash function. The result is a vector of fixed dimension, regardless of the length of the original text. This method is fast, but it can lead to a loss of quality due to possible hash collision.</p>
<h4>TFIDF</h4>
<p>It is a vectorization method that measures the importance of a word in a document compared to all the documents in a corpus. It uses two measures: the frequency of the term (TF) and the inverse of the frequency of the term in the corpus (IDF).</p>
<h4>CountVectorizer</h4>
<p>It is a vectorization method that counts the number of occurrences of each word in a document and creates a vector of numerical features as a result</p>
<h2>Classifiers</h2>
<p>These are machine learning algorithms used to predict the class membership of a sample based on its characteristics. There are several types of classifiers, such as linear regression, logistic regression, linearSVC, multinomialNB and votingclassifier.</p>
<h4>Logistic Regression</h4>
<p>It is a machine learning algorithm used to predict a binary variable from explanatory variables. It consists in modeling the probability of the target event using a logistic function.</p>
<h4>LinearSVC</h4>
<p>It is a machine learning algorithm used for binary or multi-class classification. It consists in finding an optimal separation hyperplane between the different classes of data using a cost function.</p>
<h4>MultinominalNB</h4>
<p>It is a machine learning algorithm used for multi-class classification. It uses a multinomial probability model to predict the class of membership of a sample.</p>
<h4>VotingClassifier</h4>
<p>It is a machine learning algorithm that uses the aggregation of predictions from multiple independent classifiers to predict the class membership of a sample. It can be used to improve the accuracy of the prediction by combining the strengths of different classifiers.</p>

<h2>Research methods</h2>
<h3>Search for the best combination of cleaner/vectorizer/classifier</h3>
<p>We used several methods to find the best combination of [data cleaning, vectorization, classifiers], including an algorithm that tries all combinations.</p>
<h3>Search for the optimal parameters</h3>
<p>We used <code>GridSearch</code> to find the best parameters for some classifiers.</p>
<h4>Train test split</h4>
<p>We've created a function that tries the test sizes and the optimum <code>random_state</code> for the <code>train_test_split</code> for each method.</p>