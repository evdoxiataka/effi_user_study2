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
- **LOGS**: there are 4 .csv files for each participant each containg the unique (anonymized) Prolific ID of the participant.
    - **_postquestionnaire.csv**: It contains participant's responses to post-task questionnaire
    - **_prequestionnaire.csv**: It contains participant's responses to pre-task questionnaire
    - **_training.csv**: It contains the results of integrating participant's feedback instances into the model retraining it.
    - **interactions/<ProlificID>.csv**: It contains all the interactions of the participants with the user interface. More specifically the firing of the following functions is recorded in the .csv files:
        - APPLICATIONROW\_CLICKED\_Applications\_List:  Select an application
        - OKBUTTON\_CLICKED\_DECIDE\_MODAL\_Applications\_List: Give rating and weight (if desired)
        - apply\_refine\_search: Filter for an attribute
        - REVERTBUTTON\_CLICKED: Revert effects of last provided feedback instance
        - SORTING\_CLICKED\_Applications\_List\_Predicted Decision: Sort the Applications List for predicted decision 
        - SORTING\_CLICKED\_Fairness\_Metrics\_Attribute: Sort the Fairness Metrics Table for attribute
        - SORTING\_CLICKED\_Applications\_List\_Fairness: Sort the Applications List for fairness
        - SELECT\_ATTRIBUTES\_Fairness\_Metrics: Select an attribute to show in Fairness Metrics Table

### processed_data
Various files containing processed data and some code for cleaning them can be found here. Some .csv files are retrieved from https://anonymous.4open.science/r/exploring_impact_of_lay_user_feedback_for_improving_ai_fairness/README.md.
- **checkingCollectedData.ipynb**: code for cleaning and checking users' feedback
- **feedback.ipynb**: code to create statistics about participants' interactions with the UI
- **demographics.ipynb**: code to create statistics about participants' demographics available through Prolific
- **QuestionnaireInfoInCSV.ipynb**: code to create a single csv containing all participants' responses to questionnaires.

### results
Any file resulting from the analysis of collected data organized subfolders based on the approach of integrating feedback and retraining the model can be found here.

## code
The folder "code" contains 3 folders: 

### dataset_model
Code for the AI model can be found here.

1. **AI_Model_Training.ipynb**: This notebook contains the code for training the AI Model based on the preprocessed data and predicting the outcome of the test set (i.e., the 100 loan applications shown to participants through the UI).

### analysis
This contains the code for the Analysis. The following notebooks should be run in the provided order.

1. **FeedbackIntegration_Training.ipynb**: This notebook contains the code for integrating participants' feedback, retraining the AI Model, and evaluating fairness and accuracy before and after the integration of feedback.  
2. **FeedbackIntegration_Analysis.ipynb**: This notebook contains the code for analysing the results from the integration of participants' feedback.

