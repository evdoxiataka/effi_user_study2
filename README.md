# Human-in-the-loop AI Fairness: Stakeholders Auditing and Fixing Issues Through Interactive Machine Learning
This repository constitutes a supplementary material for the paper "Human-in-the-loop AI Fairness: Stakeholders Auditing and Fixing Issues Through Interactive Machine Learning" submitted to FAccT-24.

It contains the:

- code of the AI model (XGboost) trained and used to collect feedback from real users with the aim to fix fairness issues in loan application decisions. The code includes how we preprocessed the "Home Credit Default Risk" dataset (https://www.kaggle.com/competitions/home-credit-default-risk/data), trained the model, and prepared the data shown to participants to collect feedback.
  
- code of the analysis, namely how we integrated participants' feedback and evaluated fairness and accuracy,

- collected data from participants and any code processing it.

Let's see the breakdown of the contents in this repository: 
## data
The folder "data" contains 4 folders:

### HomeCreditDataset
The following files were downloaded from https://www.kaggle.com/competitions/home-credit-default-risk/data (**Please note** that you need to download these files from this link and add them into this forlder before running any code. These files were exceeding the size limits of Github and could not be tracked.)
- **application_train.csv**: training set that includes the TARGET variable
- **application_test.csv**: test set that does not include the TARGET variable

### collected_data 
The collected data from the user study and some code for cleaning them can be found here.
- **prolific_export_demographics.csv**: this file contains the participants' demographics export from Prolific
- **LOGS**: each pair of .csv and .log files correspond to a single participants and are named after the unique (anonymized) Prolific ID of the participants. The logs contain the weights-related interactions. The .csv files contain all the interactions of the participants with the user interface. More specifically the firing of the following functions is recorded in the .csv files:
    - select\_applications:  Select an application
    - select\_reject\_application: Give rating and weight (if desired)
    - apply\_refine\_search: Filter for an attribute
    - Click on Feature Combination Button: Select Feature Combination
    - Applications\_Prediction\_Confidence: Sort for prediction confidence
    - similar\_application: View Similar applications
    - Clicked-on-Causal-Graph-Node: Select Causal Graph Node.

### processed_data
Various files containing processed data and some code for cleaning them can be found here.
- **CollectedData_Cleaning.ipynb**: code for cleaning users' feedback (remove duplicates and blank columns)
- **Feedback_final.csv**: contains all participants' feedback after cleaning

### results
Any file resulting from the analysis of collected data organized subfolders based on the approach of integrating feedback and retraining the model can be found here.

## code
The folder "code" contains 3 folders: 

### dataset_model
Code for the dataset preprocessing and the AI model can be found here. The following notebooks should be run in the provided order.

1. **Data_Preprocessing.ipynb**: This notebook contains the code for the preprocessing of the "Home Credit Default Risk" data before the training of the model.
2. **AI_Model_Training.ipynb**: This notebook contains the code for training the AI Model based on the preprocessed data and predicting the outcome of the test set (i.e., the 1,000 loan applications shown to participants through the UI).
3. **Data_CausalAnalysis.ipynb**: This notebook contains the code for the causal analysis of the test set.
4. **TestSetInfo_Preparation.ipynb**: This notebook contains the code for preparing the applications-related information for the test set that was shown to participants through the UI prototype.

### analysis
This contains the code for the Analysis. The following notebooks should be run in the provided order.

1. **FeedbackIntegration_Training.ipynb**: This notebook contains the code for integrating participants' feedback, retraining the AI Model, and evaluating fairness and accuracy before and after the integration of feedback.  
2. **FeedbackIntegration_Analysis.ipynb**: This notebook contains the code for analysing the results from the integration of participants' feedback.
3. **biasMitigation_vs_HumanFeedback.ipynb**: This notebook contains the code for calculating fairness of the model afer applying a bias mitigation algorithm (aif360 reweighing (label massaging) preprocessing).

