# E-Commerce_T5 _ Text-Only Sentiment Analysis of E-Commerce Reviews

Text-Only Sentiment Analysis of E-Commerce Reviews

This project builds a machine-learning pipeline that predicts whether an e-commerce product review is positive, neutral, or negative using the written review text. The project is based on a dataset of 500 AirPods-related reviews and demonstrates a complete natural-language-processing workflow with Pandas and Scikit-learn.

The model intentionally does not use star ratings, price, seller information, product metadata, or URLs as predictive inputs. This design keeps the experiment focused on language and reduces the risk of data leakage.

Project Objectives

The main objective is to demonstrate how raw customer reviews can be transformed into features and used to train a text-classification model. The project covers dataset inspection, text cleaning, weak-label generation, feature engineering, TF-IDF vectorization, Logistic Regression, and model evaluation.

Because the original CSV does not contain a sentiment target, the notebook creates demonstration labels from positive and negative word lists. These labels are useful for teaching and workflow demonstration, but they should not be treated as manually verified ground truth.

Dataset

The input file is airpods_cleaned.csv. It should be stored in the same directory as the notebook.

Property
Description
Domain
E-commerce product reviews
Product category
AirPods and wireless earbuds
Number of original reviews
500
Main text column
written_review
Target classes
positive, neutral, negative
Missing values
None reported in the notebook
Duplicate written reviews
25 identified before modeling
Train/test split
80% training and 20% testing




The original dataset contains additional fields such as product name, price, seller, review rating, reviewer, purchase date, purchase status, image count, and product URL. These fields are inspected but are not used as model inputs for the text-only experiment.

Sentiment Labeling Strategy

The dataset does not provide a dedicated sentiment column. The notebook therefore generates weak labels by counting words from two small vocabularies:

•
If positive words occur more often than negative words, the review receives the positive label.

•
If negative words occur more often than positive words, the review receives the negative label.

•
If the two counts are equal, the review receives the neutral label.

The resulting label distribution is:

Sentiment
Number of reviews
Positive
358
Neutral
108
Negative
34
Total
500




This distribution is imbalanced because positive reviews are much more common than negative reviews in the supplied data. The model uses balanced class weights to reduce the effect of this imbalance.


Important limitation: The evaluation scores measure how well the model reproduces the weak word-list labels. They do not prove that the model agrees with human judgments of sentiment.

Project Workflow

The notebook follows the workflow below.

1. Load and inspect the data

Pandas loads the CSV file and displays its shape, columns, and sample records. The notebook also checks for missing values and repeated written reviews.

2. Clean the review text

The written reviews are normalized into a clean_review column. The cleaning process supports consistent feature extraction and TF-IDF vectorization.

3. Remove duplicate reviews

The notebook identifies 25 duplicate written reviews. These duplicates are removed before model training to reduce the risk that the same review appears in both the training and testing sets.

4. Create engineered features

The project calculates several lightweight numerical features from each review:

Feature
Definition
Purpose
word_count
Number of words
Measures review length
character_count
Number of characters
Measures text size
unique_word_count
Number of distinct words
Measures vocabulary variety
exclamation_count
Number of exclamation marks
Captures expressive emphasis
average_word_length
Characters divided by words
Describes writing style
lexical_diversity
Unique words divided by total words
Measures vocabulary diversity
exclamation_density
Exclamation marks divided by words
Measures punctuation intensity




The final model uses the cleaned review text together with three scaled numerical features: average_word_length, lexical_diversity, and exclamation_density.

5. Convert text into numerical features

The notebook uses a Term Frequency–Inverse Document Frequency (TF-IDF) vectorizer. TF-IDF represents words numerically while reducing the influence of words that appear in many reviews. The vectorizer uses unigrams and bigrams, allowing the model to learn from both individual words and two-word phrases.

6. Train the classifier

The classifier is a Scikit-learn Logistic Regression model inside a reproducible pipeline. The pipeline applies the following operations:

1.
TF-IDF transformation for the review text.

2.
Standard scaling for the numerical features.

3.
Logistic Regression classification.

4.
Balanced class weighting to account for the uneven label distribution.

The data is divided using a stratified 80/20 train/test split with random_state=42.

Model Results

The trained model achieved the following accuracy scores:

Metric
Result
Training accuracy
95.7%
Testing accuracy
86.0%
Majority-class baseline accuracy
72.0%
Majority-class baseline macro F1
0.279
Logistic Regression macro F1
0.801




The Logistic Regression model improved test accuracy by 14 percentage points compared with the majority-class baseline. The macro F1-score also improved substantially, which indicates better performance across the three sentiment classes rather than simply predicting the most frequent class.

The difference between training accuracy and testing accuracy suggests some degree of overfitting. The test score is therefore more informative than the training score when estimating performance on unseen reviews.

Example Prediction

The notebook also tests the trained model on previously unseen sample reviews. This demonstrates how a new review can be cleaned, converted into the same numerical features used during training, and passed through the trained pipeline.

The example predictions also show why aggregate metrics should be interpreted carefully. A clearly negative review can still be predicted as positive when the model relies heavily on individual words and does not fully understand negation, context, or phrase meaning.

Limitations

This project is primarily an educational demonstration. Its results should not be interpreted as a production-ready sentiment system.

The most important limitation is the use of weak labels. The labels were created from a small word list rather than from human annotation. This means that the model may learn the word-list rules instead of learning human sentiment judgments.

The dataset is also relatively small and imbalanced. It contains only 34 weakly labeled negative reviews, which limits the model's ability to learn the full range of negative language. Reviews may also contain informal spelling, Roman Urdu, mixed languages, sarcasm, negation, and product-specific expressions that simple TF-IDF features cannot fully interpret.

The notebook identifies duplicate reviews before modeling, but a larger study should also examine near-duplicate reviews, templated seller text, and possible review-generation artifacts. A single train/test split may also produce an optimistic or unstable estimate of generalization performance.

Recommended Improvements

A stronger version of the project could:

1.
Replace weak word-list labels with human-verified sentiment annotations.

2.
Increase the size and linguistic diversity of the dataset.

3.
Use cross-validation and class-specific evaluation metrics.

4.
Add negation-aware preprocessing, such as recognizing phrases like not good.

5.
Compare Logistic Regression with models such as Linear Support Vector Machines or gradient-boosted classifiers.

6.
Evaluate multilingual and Roman Urdu reviews separately.

7.
Use transformer-based language models when a larger labeled dataset and additional computational resources are available.

8.
Perform systematic error analysis by sentiment class, review length, language, and product type.

Requirements

The notebook requires Python 3 and the following packages:

Bash


pip install numpy pandas matplotlib scikit-learn jupyter



How to Run

1.
Place sentiment_analysis.ipynb and airpods_cleaned.csv in the same directory.

2.
Install the required Python packages.

3.
Start Jupyter Notebook or JupyterLab:

Bash


jupyter notebook





4.
Open sentiment_analysis.ipynb.

5.
Run the notebook cells from top to bottom.

The notebook expects the CSV file path to be:

Python


csv_path = "airpods_cleaned.csv"



If the CSV is stored elsewhere, update this path before running the data-loading cell.

Project Files

File
Description
sentiment_analysis.ipynb
Main notebook containing the complete analysis and model workflow
airpods_cleaned.csv
Required input dataset referenced by the notebook
README.md
Project documentation and usage instructions




Conclusion

This project demonstrates an end-to-end text-classification workflow for e-commerce reviews. The TF-IDF Logistic Regression model achieved 86.0% test accuracy and a macro F1-score of 0.801, outperforming a majority-class baseline. The results show that written language contains useful sentiment signals, even when explicit star ratings are excluded.

At the same time, the project makes an important methodological distinction: strong performance against weak labels is not the same as validated performance against human sentiment judgments. Human-verified labels, a larger dataset, and more advanced language-aware modeling would be the appropriate next steps for a reliable sentiment-analysis system.

References

[1] Pandas Documentation
[2] Scikit-learn Text Feature Extraction Documentation
[3] Scikit-learn Linear Models Documentation
[4] Scikit-learn Model Evaluation Documentation
[5] Scikit-learn Pipeline API Reference
