# CSE151A_GP
Group Project for CSE151A
 
# The Dataset we are using:
https://www.kaggle.com/datasets/chaoticqubit/nuuly-customer-reviews-second-hand-apparels

Dataset of customer reviews on second hand apparels. It includes features such as customer's body type and preferences, and the star rating of the review.

# Our Jupyter Notebooks:
- Pre Processing and EDA notebook
[![CSE151A_GroupProject.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/CSE151A_GroupProject.ipynb)]
- Model 1 notebook
[![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/model.ipynb)]

# Data Preprocessing Steps
- Our goal is to predict review star rating based on review features including both product features and user features.
- Our data contains some products that have no reviews. We chose to drop these products from our dataset
- For feature selection we used both feature selection methods such as correlation coefs and our intuition for what features will be important.
- Data Transformations:
  - One Hot Encode categorical features like:
    - Size: 'S','M', 'L', etc.
    - Body-Type: 'Apple', 'Pear', etc.
    - Color: 'Red', 'Green', 'Blue', etc.
  - Convert features to numerical:
    - height: '5'3' to 63 inches
    - weight: '130lbs' to 130
    - age: '51' to 51
  - Text Processing:
    - We processed the text using nltk to help us tokenize our review text
 - We created an initial train test split to use for our first model's training and testing
    - We may later split our data into train, test, and validate splits to help us validate more complex models
   
# Our First Model
- For our first model we plan to fit a simple linear regression to see how a baseline model performs
- We will use this model to compare a simple strategy to our future more complex models
- Since our task is classification we chose to bin our model's predictions to the nearest int (1-5)

## First Model Performance:
### Classification Metrics
  - Accuracy: 0.3073196054680741
  - Precision: 0.5725372565028342
  - Recall: 0.3073196054680741

### Non Classification Metrics
  - Train MAE: 0.7380
  - Test MAE: 0.7387
  - Train MSE: 0.878
  - Test MSE: 0.8817
  - Train R^2: 0.3914
  - Test R^2: 0.3902

## First Model Fit:

![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/f5bd7752-dcb6-435d-a2da-3e7b65baaaa9)

![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/b5fe68a1-75d8-47a2-948d-d2b1e97cc0f5)

# Future Models
- Random Forest
  - Random Forest models are naturally capable of multi-class classification and are strong at preventing overfitting
- Neural Network
  - Neural networks are also naturally capable of multi-class classifications and can perform strongly depending on our chosen activation and loss functions.
 
# Milestone 3 Conclusion
- Overall our linear regression performed poorly as a review rating predictor
  - We wanted to get a quick idea of how a simple model would perform on this classification task
  - Realisitically we shouldn't have chosen a linear regression as our simple model as we have a classification task not regression
  - Some rating predictions went into the negatives, which is not realistic
  - We binned our linear regression's predictions to make it perform like a classifier
  - Instead we likely could have implemented a one-vs-rest multiclass logistic regression as a simple classifier
  - Our model still gave us a decent foundation to work off after being binned to perform like a classifier
  - In terms of the fitting graph, it is **underfitting**. This model is too simple shown by the poor accuracy, precision, and recall, and regression on a classification problem can not capture the pattern of predicting a review. 
- To improve our rating predictor we plan to do a few things
  - Implement review title and text sentiment as a feature
    - Sentiment is likely one of the strongest features in this dataset and as a human reader it is what we would use to predict rating
  - Switch model types
    - While linear regression is a good baseline model for understanding the performance of future models it is not especially strong for this specific classification task
    - Instead we would look to use a model that is naturally capable of multi-class classification due to star rating being an integer from 1 to 5.
    - At the moment we plan to look into implementing a random forest model and a neural network as both can be used for strong multi-class classifiers.
  - With future models we would look to perform hyperparameter tuning to boost model performance.
  - We may have to balance our dataset by star rating so that we don't have an imbalance such as 70% of reviews being 5 star.

# Milestone 4:

In this milestone you will focus on building your second model. You will also need to evaluate your this model and see where it fits in the underfitting/overfitting graph.

1. Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.
- 
2. Train your second model

3. Evaluate your model compare training vs test error

4. Where does your model fit in the fitting graph, how does it compare to your first model?

5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

5. What is the plan for the next model you are thinking of and why?

6. Update your readme with this info added to the readme with links to the jupyter notebook!

7. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?

Please make sure your second model has been trained, and predictions for train, val and test are done and analyzed. 
