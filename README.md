# Credit Risk Classification

In this project, I’ll use various techniques to train and evaluate a model based on loan risk. I’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Overview of the Analysis

I used Logistic Regression Model to identify the creditworthiness of borrowers.

- DataFrame is created from the lending_data CSV file given on this project. Data consisted on 77536 rows and 8 columns.
- y is 'loan_status'. 0 describes a 'healthy loan' while 1 describes a 'high risk loan'. 
- Features (X) is all the remaining columns from the dataframe.

Stages of Machine Learning Process on This Analysis:
1. Read the data and identified X and y values.
2. Split the dataset into Training and Testing sets by using train_test_split.
3. Create a Logistic Regression Model with the original data
4. Generate a confusion matrix and the classification report


## Results

### Machine Learning Model 1: Logistic Regression
**Classification Report Results**
![classification_report_lr](https://github.com/skythelimitdt/credit-risk-classification/blob/main/Resources/classification_report_lr.png)

- **precision:** The model achieved a perfect 100% precision for healthy loans, while high-risk loans had an 87% precision, meaning 13% were false positives.
- **recall:** It correctly identified 100% of healthy loans and 95% of high-risk loans, leaving 5% as false negatives.
- **f1-score:** The F1-score of 1 for healthy loans indicates an excellent balance between precision and recall, while 0.91 for high-risk loans demonstrates a strong but slightly less balanced performance.
- **accuracy:** The model correctly classified 99% of all samples.

Overall, this model performs well, demonstrating high precision and recall for both loan categories. With a 99% accuracy rate, it is highly reliable, though there is room for improvement, particularly in reducing the 13% false positive rate for high-risk loans.

### Machine Learning Model 2: Support Vector Model

**Classification Report Results**
![classification_report_svc]([url](https://github.com/skythelimitdt/credit-risk-classification/blob/main/Resources/classification_report_svm.png)

The SVM model produced results similar to Logistic Regression, with few key differences:
- **Higher recall (98%),** meaning it reduced the number of false negatives compared to Logistic Regression.
- **Higher f1-score (92%),** indicating a better balance between precision and recall.
- **Higher macro-average score (99%)** indicating a better average score for high risk loans for precision, recall, and f1-score.


## Summary Analysis and Model Recommendation

Both the **Logistic Regression** and **Support Vector Machine (SVM)** models performed well in classifying loan risk, achieving high accuracy, precision, recall, and f1-scores. However, key differences distinguish their performance:

1. **Accuracy** – Both models correctly classified **99% of total samples**, indicating strong reliability.
2. **Precision** – Logistic Regression had a **100% precision for healthy loans** but an **87% precision for high-risk loans**, meaning 13% of high-risk loans were misclassified as low risk (false positives).
3. **Recall** – SVM outperformed Logistic Regression in recall for high-risk loans **(98% vs. 95%)**, reducing false negatives and ensuring more high-risk loans were correctly identified.
4. **Balance of Metrics** – SVM had a higher macro-average score (96%), indicating a better average of precision, recall, and F1-score across both loan categories.

### Recommended Model: Support Vector Machine (SVM)
The SVM model is recommended as the better option due to its higher recall and better overall balance in classifying both healthy and high-risk loans. SVM’s higher recall rate reduces the risk of falsely classifying high-risk loans as low risk**—a critical factor in financial decision-making**. The improved macro-average score also suggests that SVM maintains a stronger equilibrium between all evaluation metrics.

For real-world loan assessments, reducing false negatives (high-risk loans misclassified as healthy loans) is very important, making SVM the superior choice for this task.

## Technologies and Tools Used
- Jupyter Lab
- pandas, numpy
- sklearn library

## Resources and Support
ASU Bootcamp Activities
