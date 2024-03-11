# CSE151A_GP
Group Project for CSE151A
 
# The Dataset we are using:
https://www.kaggle.com/datasets/chaoticqubit/nuuly-customer-reviews-second-hand-apparels

Dataset of customer reviews on second hand apparels. It includes features such as customer's body type and preferences, and the star rating of the review.

# Our Jupyter Notebooks:
- Pre Processing and EDA notebook
[![CSE151A_GroupProject.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/CSE151A_GroupProject.ipynb)
- Model 1 notebook
[![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/model.ipynb)
- Model 2 notebook
[![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/model2.ipynb)
- Model 3 notebook
[![model.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OrenKGit/CSE151A_GP/blob/main/model3.ipynb)
  
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

# Our Second Model

## Preparation
- We found that our data was sufficient to build a second model.
- Our loss function naturally changed due to the model no longer being a linear regression but a logistic regression

## Second Model Performance

### Test Classification Report
- We have an accuracy of 0.59 which is much better than our previous model's accuracy of 0.3!
<img width="463" alt="image" src="https://github.com/OrenKGit/CSE151A_GP/assets/91357838/efc8f9c4-8294-4e01-9eb5-708fa9fe5ed3">

### Train/Test Error
Train MAE: 0.5896
Test MAE: 0.5973
Train MSE: 1.095
Test MSE: 1.1069
Train R^2: 0.2410
Test R^2: 0.2344

### Overfitting/Underfitting Graph
- From these confusion matrix graphs alongside the model performance metrics we can clearly see that the model is not overfitting
- It is possible our model may be slightly underfitting as our model's train and test performance are very similar
- If the model performance were lower we would lean towards saying it is underfitting
- We believe our model is neither overfitting or underfitting

![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/2ba984af-a216-4fca-8ef7-6676db20e1fa)
![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/36331bb5-9229-453a-a149-7a5105a1bd75)

## Hyperparameter tuning
- We chose to not do significant hyperparameter tuning for this model
- Our reasoning was that a logistic regression model doesn't really have many hyperparameters to tune anyways
- Instead of using a grid search to iterate through parameters we manually tried a few different 'solvers' for the logistic regression
- The model trains very fast so this is easy to evaluate
- We then chose to best solver for our model

## Our Next Model
- For our next model we plan to implement a random forest classifier
- We plan to implement hyperparameter tuning for our final model with gridsearchcv in order to optimize our final model
- Random forest is a more complex model that is naturally capable of multiclass classification tasks
    - It is also naturally good at resisting overfitting
    - This means that our hyperparameter tuning hopefully won't lead us to an overfit model

# Milestone 4 Conclusion
- Overall this model was relatively simple but a much better choice than our first model
  - Our first model was a decent baseline to compare to but overall a poor choice for this type of problem
  - Our second model being capable of naturally performing multiclass classification means it was much more suitable
  - Our second model actually performs reasonably well with an accuracy of 0.6, especially compared to the first model which had an accuracy of 0.3
- Our model appears to be neither overfitting or underfitting
  - Model is clearly not overfitting as the train metrics are not significantly higher than test metrics
  - Our model may exhibit mild underfitting but overall train and test perform similarly and not too poorly
  - This leads us to believe we are neither underfitting or overfitting
- This model being a logistic regression means there aren't many hyperparameters to tune
  - We did basic manual hyperparameter tuning by testing a few different solvers for the model
  - We chose the best solver and used it for our real model
- To further improve this model we likely need to fix the class imbalance
  - 5 star ratings are far more represented within the data and may lead our model to be biased towards predicting 5
- For our final model we plan to use a more complex model and optimize it through hyperparameter tuning
  - Our model will likely be a random forest
    - Random forest is naturally capable of classification tasks
    - It is also good at resisting overfitting
    - This means that our hyperparameter tuning hopefully won't lead us to an overfit model

