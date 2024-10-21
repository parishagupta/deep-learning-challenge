# deep-learning-challenge

# Alphabet Soup Charity Funding Predictor

## Project Overview

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Two machine learning models, Deep Neural Network and Random Forest, are use to predict whether a charity funding applicant will be successful based on the provided application data. Application data was provided by the Foundation and contains 34,000+ records of previous donation recepiants. 


## Models Used
1. **Deep Neural Network (DNN)**:
   - Deep learning is a subset of machine learning methods that utilize neural networks for representation learning. It takes inspiration from biological neuroscience and is centered around stacking artificial neurons into layers and "training" them to process data.
   - *Accuracy: 72.71%*
   
2. **Random Forest**:
   - Random Forest traditional machine learning model used for comparison which combines the output of multiple decision trees to reach a single result.
   - *Accuracy: 72.98%*

Based on the results, the Random Forest model slightly outperformed the Deep Neural Network model.

## Dataset
The dataset includes the following features:
- **EIN** and *Name**: Identification columns.
- **Application Type**: Type of application submitted.
- **Affiliation**: Affiliated sector of industry.
- **Classification**: Government organization classification.
- **Income Amount**: Income level of the applicant.
- **Special Considerations**: Special considerations requested.
- **Funding Amount Requested**: Amount of funding requested.

The target variable is `IS_SUCCESSFUL`, which indicates whether the funding was used effectively by the applicant.

## Project Workflow

### 1. Data Preprocessing:
- Dropped unnecessary columns such as `EIN` and `NAME`.
-For columns with 10+ unique values, determined the number of data points for each unique value
- Encoded categorical variables using `pd.get_dummies`.
- Grouped rare categorical values into a new category called 'Other' based on a chosen cutoff point.
- Scaled the data using `StandardScaler`.

### 2. Model Training:
- **Neural Network**: Two  layers were used to capture complex patterns, with ReLU activation for the hidden layers and sigmoid activation for the output layer.
- **Model Optimization**: The goal was to achieve a target predictive accurarcy of 75%+ by using TensorFlow to optimize the existing model. Various techniques were applied, including adjusting the number of neurons, layers, and epochs to improve performance. Despite optimization attempts, the accuracy remained similar at 72-73%.

### 3. Model Evaluation:
- The models were evaluated on test data. The Neural Network performed slightly better with a 72.71% accuracy, while the Random Forest achieved 71.0% accuracy.
- The Random Forest model was used as an alternative to the Neural Network. Although the Random Forest performed slightly worse, it may be more suitable for simpler datasets or when faster training time is needed.

## Libraries used
- TensorFlow
- Scikit-learn
- Pandas

## Files Included
- **AlphabetSoupCharity.ipynb**: Jupyter notebook containing data preprocessing, model training, and evaluation.
- **AlphabetSoupCharityOptimization.ipynb**: Jupyter notebook containing 3 iterations of the above model to attempt to achieve a target predictive accuracy of 75%+.
- **AlphabetSoupCharity.h5**: The saved Neural Network model in HDF5 format.
- **AlphabetSoupCharityOptimization_nn().h5**: nn1, nn2, and nn3 are the 3 iterations of the original model.

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
 
