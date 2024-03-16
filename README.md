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

# Multiple Machine Learning Models’ Analysis of Customer Reviews on Second-Hand Apparel
# Introduction
Online commerce and retails platforms have become crucial in recent times with websites such as Amazon and eBay. Buying clothes and apparel online have gained increasing popularity, and most young adult Americans prefer online shopping over in-person. With today’s inflation and increased prices however, buying used clothes or even renting them have become more prominent, known as thrifting. Being second-hand however, customer rating and approval becomes more crucial than buying the typical new clothes. 

Through machine learning and natural language processing, companies can improve their products by predicting the sentiment of their reviews through analyzing the characteristics and features of a review that will lead to more sales. Other important uses for clothing stores can be quality control of products, insight on current trends in fashion or styles, and better product recommendations through sentimental analysis. 

# Methods
## Data exploration
The dataset is from kaggle, who’s creator used a web scraping tool on nuuly.com to obtain reviews.

Each entry consisted of the characteristics of each customer like their size and age. The characteristics are as follows:
- product_link: Website link to the product
- product_name: Name of the clothing product
- product_description: Website’s description of the product
- product_price: Price of product
- review_posted_by_username: Username of whoever posted the review
- user_size: Relevant clothes size of the user (ex. XS to XL)
- user_color: Color selected of the product that the reviewer brought
- user_height: Height of user, in feet and inches
- user_weight: Weight of the user, in pounds
- user_body_type: “Shape” of the body (ex. hourglass, apple)
- user_bra_size: American metric of bra size
- user_age: Age of user
- review_date: Date review was posted
- review_title: Title of the review posted
- review_text: The full text of the review.
- star_ratings: The rating, 1-5, of the review
Number of observations: 3422
Number of reviews any features missing or null: 335
Exploratory techniques such as correlation matrices and scatterplots on the multiple selected features were done to check the outcomes
## Preprocessing
(Notebooks: CSE151A_GroupProject.ipynb, model.ipynb)
The original data was cleaned up by dropping null or empty reviews, creating an average star rating for each product based on the number of reviews for that product, and creating better formatting the .json file for easier extraction. For example, the format included assigning all reviews to a product name rather than having to check every single review.  Conversions of proper variable types were also done, such as height and weight being converted from their string format into integers of pounds and inches respectively. One hot encoding is useful for models with feature selection models, and was used on user size, body type, bra size, and color. Processing the text of the reviews was done through natural language processing and tokenizing each word and associating them to their rating.
### Model 1
Linear Regression

The linear regression model predicts the star rating from one to five by using the nearest integer based on the text of the review. The reviews are processed through bag-of-words, which keeps track of the frequencies of each word appearing and predicts a star rating based off of the words. Then the actual star ratings and model predictions are compared with the following graph below. 
### Model 2
Multinomial Logistic Regression

Multinomial logistic regression model does the same in that it predicts the star rating to the nearest integer based on the review text. Bag-of-words counts the frequency of the words, and the model uses LBFGS as the solver with a maximum of 1000 iterations for optimal predictions. Multiple solver types were tested as a form of hyperparameter tuning, with LBFGS ultimately giving the best accuracy. 
### Model 3
Random Forest Classification

The random forest classifier tokenizes and processes the text through the natural language toolkit and its tools such as lemmatization and stop words. Bag of words is used, and the model itself uses 100 decision trees as its estimators. Grid search is used for hyperparameter tuning and finds the best parameters along with the best score. Our grid search parameters were defined as: 

param_grid = { 
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4], 
    'max_features': ['auto', 'sqrt'], 
    'bootstrap': [True, False],   
    'random_state': [42],
}

# Results
## Model 1
Classification Metrics
Accuracy: 0.3073196054680741
Precision: 0.5725372565028342
Recall: 0.3073196054680741
Non Classification Metrics
Train MAE: 0.7380
Test MAE: 0.7387
Train MSE: 0.878
Test MSE: 0.8817
Train R^2: 0.3914
Test R^2: 0.3902

![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/f5bd7752-dcb6-435d-a2da-3e7b65baaaa9)
![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/b5fe68a1-75d8-47a2-948d-d2b1e97cc0f5)

## Model 2
Accuracy: 0.59
Weighted average precision: 0.55
Weighted average recall: 0.59
Train MAE: 0.5896 
Test MAE: 0.5973 
Train MSE: 1.095 
Test MSE: 1.1069 
Train R^2: 0.2410 
Test R^2: 0.2344

![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/2ba984af-a216-4fca-8ef7-6676db20e1fa)
![image](https://github.com/OrenKGit/CSE151A_GP/assets/91357838/36331bb5-9229-453a-a149-7a5105a1bd75)

## Model 3
Best grid search parameters: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200, 'random_state': 42}
Best accuracy: 0.6473

<img width="428" alt="image" src="https://github.com/OrenKGit/CSE151A_GP/assets/91357838/f4ee4f28-a794-4734-89b7-f4a1fc1058b4">
<img width="803" alt="image" src="https://github.com/OrenKGit/CSE151A_GP/assets/91357838/6d7e9d24-4665-4bad-a577-a40eb1e268f7">
<img width="817" alt="image" src="https://github.com/OrenKGit/CSE151A_GP/assets/91357838/70972443-09bf-4e2f-a353-fee912f11cec">

# Discussion
Ultimately a lot of data preprocessing ended up not being used, but was good for information and data exploration. The features such as the review user’s weight, height, and shape or the color and size of clothes weren’t used due to us deciding that all models should use sentimental analysis. This was for easier model comparison and seeing how each model method performs
Linear regression was used as the initial model due to its simplicity. Since the task is classification but our linear regression output decimals we had to threshold our output values to star rating integers. While linear regression is not the best model to do classification, starting with a regression model will help better understand the relationship between the reviews and the rating, providing important information for selecting future models. 
Since the linear regression model performed poorly as a review rating predictor, we chose multinomial logistic regression model as our second model because it’s well-suited for classification of predicting star ratings from 1 to 5 based on reviews. We also intended to compare this model with the linear regression model to assess how much our model has improved by incorporating nonlinear relationships into our predictions. This resulted in a large boost in accuracy through just model choice alone. We did some basic hyperparameter tuning through manually testing different logistic regression solvers and then chose the one that performed the best on our test set. The end result was an increase from an accuracy of around 0.3 with the linear regression model to an accuracy of around 0.6 with the logistic regression.
We chose the random forest classifier as our final model because it employs an ensemble learning method, which allows the model to combine the predictions of multiple individual decision trees to improve performance. Therefore, it tends to have lower variance, greater robustness to overfitting, and better generalization performance compared to logistic regression. Since the random forest classifier also has many hyperparameters that can be tuned to boost final model performance we chose to use sklearn’s GridSearchCV to iterate through our chosen hyperparameters. The parameters we tuned included: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap. Hyperparameter tuning was the longest part of the modeling process with our chosen parameter grid taking around 40 minutes to run. Our hyperparameter tuning also only resulted in an accuracy increase of 0.01, suggesting we either were unable to find the utmost optimal hyperparameters or that optimal hyperparameters only resulted in small performance improvements. Despite this, the tuning was valuable as with our final model we wanted to get every bit of performance out of it that we could.

# Conclusion
In conclusion, while all three models performed moderately well, the Random Forest classifier achieved the highest accuracy (0.64) compared to Logistic Regression (0.59) and Linear Regression (0.31). This suggests that more complex models are necessary to capture the nuances of sentiment expressed in reviews, despite the Logistic Regression model showing a significant improvement in accuracy from the baseline Linear Regression model. However, all three models show room for improvement in predicting exact sentiment scores, as indicated by their MSE. 

There are a few more steps we believe we could take in order to achieve higher accuracy. First, we could incorporate other features outside of sentiment analysis input for our models. Despite all the data preprocessing for the features, our models only used the review text for sentiment analysis. We could choose to include features from either the product or the review including; product price, product size, user’s age, weight, height, and more. Second, with these new features we could expand our data preprocessing to include more optimal scalings of our features and possibly perform PCA to select our best features. Lastly, we could run more extensive hyperparameter-tuning on our random forest model, hoping to find more optimal hyperparameters. 

Alternatively, we can apply more advanced machine learning algorithms such as gradient boosting machines or deep learning models,which are better at handling complex tasks. We considered using a neural network for our final model and if we had the chance it would have been interesting to see how it compares to the random forest performance with both being more complex models. 

Overall, we succeeded in developing a decently performant final model by iterating on each of our previous models and understanding what could be improved with each. 

# Collaboration
Juheng Wu - Coder - Researched for possible models, built the linear regression model, logistic model, bag of words, and helped with other data preprocessing.

Oren Kaplan -  Organizer/Writer - Write up for Milestone 2, 3, 4, and Final. Data preprocessing for Milestone 2, Hyper Parameter tuning for Random Forest. Model performance graphs for Linear Regression and Random Forest models. Communicated with the team to set up meetings/deadlines and with staff to receive new dataset.

Vladimir Em  - Organizer/Coder - Coding for Random Forest Model and Hyperparameter Tuning.  Model performance graphs for Linear Regression and Random Forest models. Communication for deadlines and meetings.

Erik Pak  - Organizer/Editor - Setup meetings and hosted Vscode liveshare sessions for the group. Helped with readme (proofreading/editing and jupyter notebook links), code for milestone 2, and visualizations of model performance. Did Milestone 1 although ultimately not used.

Derek Ly  -  Writer - Did checkups of requirements, gave feedback, and other finishing touches for each checkpoint’s README. Wrote the introduction and methods section of the final writeup. Also some cleanup on model 3 

Beomsuk Seo  - Coder - Helped with abstract for milestone 1 and performed data preprocessing for milestone 2. Hosted VSCode live sessions for the group.

Yuhao Zhang - Writer - Helped with preprocessing code for Milestone 2. Contributed to describing the models and writing the conclusion for the final write-up.
