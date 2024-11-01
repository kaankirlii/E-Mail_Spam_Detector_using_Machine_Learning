# E-Mail_Spam_Detector_using_Machine_Learning
This project demonstrates a machine learning approach to classify emails as spam or not spam using a Naive Bayes classifier. It leverages Natural Language Processing (NLP) techniques to convert email text into numerical features, making it a straightforward and practical application for beginner to intermediate Data Scientists.

## Project Goals
- **Objective**: To build a model that classifies emails as either spam or not spam based on their content.
- **Dataset**: A sample dataset of 500 emails was created with random spam and non-spam email content.
- **Target Audience**: This project is intended for senior Data Scientist employers or anyone looking to develop skills in NLP and binary classification.

## Project Overview
This project walks through:
1. **Data Preparation**: Creation of a dataset containing 500 emails labeled as `spam` or `not spam`.
2. **Data Processing**: Using Count Vectorizer to transform text data into numerical form.
3. **Modeling**: Training a Naive Bayes classifier for binary classification.
4. **Evaluation**: Measuring model performance with accuracy and classification reports.

## File Structure
- `email_spam_detector.ipynb`: Jupyter Notebook containing all code, explanations, and output.
- `data`: Contains sample email data used in this project.
- `README.md`: Project documentation.

## Getting Started

To run this project on Google Colab or locally, ensure you have the required libraries installed.

### Prerequisites
- Python 3.x
- Libraries:
  ```bash
  pip install pandas numpy scikit-learn
  ```

### Running the Project
1. Clone this repository or download the `email_spam_detector.ipynb` file.
2. Open the notebook in Google Colab or a Jupyter Notebook environment.
3. Run each cell to process data, train the model, and view results.

## Dataset
This project uses a sample dataset with 500 rows of randomly generated email texts, divided evenly between spam and non-spam labels. Below is an example of the dataset structure:

| Email Text                                                                                         | Label     |
|----------------------------------------------------------------------------------------------------|-----------|
| "Congratulations! You've won a $1,000 gift card. Click here to claim now!"                         | spam      |
| "Meeting is scheduled at 3 PM tomorrow, please confirm."                                           | not spam  |
| "Last chance to win a trip to Hawaii!"                                                             | spam      |
| "Hello, wanted to check if you're available for a quick call."                                     | not spam  |

The emails cover various patterns found in spam and non-spam messages, providing the model with basic yet distinct training data.

## Code Walkthrough

### Step 1: Data Preparation
- We create a dataset of 500 emails, labeled as either `spam` or `not spam`.
- Labels are assigned randomly, with equal distribution for model balance.

### Step 2: Data Processing
- Using `CountVectorizer` to convert email text into numerical features suitable for Naive Bayes classification.

### Step 3: Model Training
- We use a Naive Bayes classifier due to its efficiency and effectiveness for text classification.

### Step 4: Evaluation
- The model is evaluated using accuracy and a classification report to assess precision, recall, and F1-score.

## Results
- **Accuracy**: Achieved around 80-85% accuracy on the test dataset.
- **Classification Report**: Shows the precision, recall, and F1-score for each class.

## Sample Code
Here's a snippet showing the model training and evaluation process:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Future Improvements
- **Expand Dataset**: Increase the dataset size with more diverse email texts.
- **Advanced Models**: Experiment with other algorithms, such as Support Vector Machines or Random Forest.
- **Feature Engineering**: Apply techniques like TF-IDF for more nuanced text feature representation.

## License
This project is open-source.

## Contributions
Feel free to fork this repository, submit issues, and open pull requests to enhance the project.

## Acknowledgments
Thanks to the open-source community and various resources that made this project possible. This project is designed to be a practical example for those looking to enter the field of NLP and binary classification with machine learning.

