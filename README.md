# CSE151A_GP
Group Project for CSE151A
 
# The Dataset we are using:
https://www.kaggle.com/datasets/uciml/student-alcohol-consumption

# Data Preprocessing Steps
# We need to rewrite this section
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
    - We will likely implement review title and text processing using a library to more easily tokenize and process text sentiment
   
# Our First Model
- For our first model we plan to fit a simple linear regression with no sentiment analysis to see how a baseline model performs
- We will use this model to compare a simple strategy to our future more complex models

# Future Models
- Random Forest
  - Random Forest models are naturally capable of multi-class classification and are strong at preventing overfitting
- Neural Network
  - Neural networks are also naturally capable of multi-class classifications and can perform strongly depending on our chosen activation and loss functions.
 
# Milestone 3 Conclusion
- Overall our linear regression performed poorly as a review rating predictor
  - **Regression performance stats**
  - ...
- To improve our rating predictor we plan to do a few things
  - Implement review title and text sentiment as a feature
    - Sentiment is likely one of the strongest features in this dataset and as a human reader it is what we would use to predict rating
  - Switch model types
    - While linear regression is a good baseline model for understanding the performance of future models it is not especially strong for this specific classification task
    - Instead we would look to use a model that is naturally capable of multi-class classification due to star rating being an integer from 1 to 5.
    - At the moment we plan to look into implementing a random forest model and a neural network as both can be used for strong multi-class classifiers.
  - With future models we would look to perform hyperparameter tuning to boost model performance.
  - We may have to balance our dataset by star rating so that we don't have an imbalance such as 70% of reviews being 5 star.
 
# Redoing Milestone 2
- Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, plot example classes of the image, size of images, are sizes uniform? Do they need to be cropped? normalized? etc.
- Plot your data. For tabular data, you will need to run scatters, for image data, you will need to plot your example classes.
- How will you preprocess your data? You should explain this in your readme.md file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.

# Milestone 3
In this milestone you will focus on finishing any preprocessing and build your first model. You will also need to evaluate your this model and see where it fits in the underfitting/overfitting graph.

1. Finish major preprocessing
     a. For tabular data, this includes scaling and/or transforming your data, imputing your data, encoding your data, feature expansion, Feature expansion (example is taking features and generating new features by transforming via polynomial, log multiplication of features).

     b. For image data this includes scaling the number of pixels from HD to SD i.e. down sampling (1280x720x3RGB to 640x360x3RGB or further). This also includes flattening your image like we did in HW2 64x64 to 4096 x 3RGB so 12888. Another option is transforming RGB to gray scale.
   
3. Train your first model

4. Evaluate your model compare training vs test error

5. Where does your model fit in the fitting graph.

6. What are the next 2 models you are thinking of and why?

7. Update your readme with this info added to the readme with links to the jupyter notebook!

8. Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

Please make sure preprocessing is complete and your first model has been trained, and predictions for train, val and test are done and analyzed. 