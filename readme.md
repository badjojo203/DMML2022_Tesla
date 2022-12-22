# Unil Kaggle competition
## Team Tesla

**Members**: 
- Kabasa Jonathan 
- Osorio Santos Joël

<h1>Detecting the Difficulty Level of French Texts</h1>
<p>The project consists in predicting the difficulty level of a text in French (A1, A2, B1, B2, C1, C2). For this purpose, we have a training dataset including the columns "sentences" and "difficulty", as well as a dataset including only the column "sentences" for which we have to predict the difficulty using different approaches.</p>
<h2>Data sets</h2>
<ul>
  <li>Training data set: includes "sentences" and "difficulty" columns</li>
  <li>Unlabelled data set: includes only the "sentences" column for which we have to predict the difficulty</li>
</ul>
<h2>Approaches tested</h2>
<ol>
  <li>Data Cleaning
    <ol>
      <li>Removal of empty words</li>
      <li>Data cleaning</li>
    </ol>
  </li>
  <li>Vectorization
    <ol>
      <li>Hash</li>
      <li>TF-IDF</li>
      <li>CountVectorizer</li>
    </ol>
  </li>
  <li>Classifiers
    <ol>
      <li>Linear Regression</li>
      <li>Logistic regression</li>
      <li>LinearSVC</li>
      <li>MultinomialNB</li>
      <li>VotingClassifier</li>
    </ol>
  </li>
<p><br><br></p>
<p>For more information about the approach and exemple of usage,<a href="documents/getstarted.md"> click here </a></p>
<p>For more information about the glossary <a href="documents/lexique.md"> click here </a></p>
</ol>
<h2>Results</h2>

<img src="documents/results.png">

[//]: # (Which is the best model?)

[//]: # (Show the confusion matrix.<img src="documents/ConfusionMatrix.jpg">)

[//]: # (Show examples of some erroneous predictions. Can you understand where the error is coming from?)

[//]: # (Do some more analysis to better understand how your model behaves.)


<h2>Link to the Video</h2>

<a href="https://www.youtube.com/watch?v=b0d5uGUHQZg">click here</a>
