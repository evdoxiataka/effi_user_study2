from utils.fairness_metrics import group_fairness, consistency_score_, counterfactual#, theil_index_, DPR_AOD_fairness
from utils.utils import add_cma_data

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

from timeit import default_timer as timer 

def reweighing(y_train, n_feed, fs):
    """
    y_train: array/list/pd.Series of labels
    n_feed:  the number of feedback instances in the training set 
             (the last n_feed labels in y_train are instances from 
             participants feedback)
    fs:      feedback significance %         
    """
    ## weights to balance classes
    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y = y_train
    )
    # print('classes_weights', classes_weights)
    
    if n_feed:
        ## weights to account for feedback
        feedback_mass_weight = len(y_train)*fs
        training_mass_weight = len(y_train)*(1-fs)
        # print('mass_weight', feedback_mass_weight,training_mass_weight)
        feedback_inst_weights = [feedback_mass_weight/n_feed]*n_feed
        training_inst_weights = [training_mass_weight/(len(y_train)-n_feed)]*(len(y_train)-n_feed)
        feedback_weights = np.array(training_inst_weights+feedback_inst_weights)
        # print('feedback_weights', feedback_weights)
    else:
        feedback_weights = [1.]*len(y_train)
    
    return classes_weights*feedback_weights

def train_and_evaluate_model(X_train_original, y_train_original,
                             X_test_original, y_test_original, 
                             train_df_test_bin, feature_weights,
                             sensitive_attrs,
                             n_feed, fs, iteration, p_id):
    """
        feature_weights: List of probabilities corresponding to each feature
        iteration: Index to participant's feedback instances (these are taken in increasing timestamps)
        p_id: Participant's ID
    """
    ## train model on original training data set
    classes_weights = reweighing(y_train_original, n_feed, fs)    
    if len(feature_weights)==0:
        model = XGBClassifier(random_state = 15, eta = 0.3)
        model.fit(X_train_original, y_train_original, sample_weight=classes_weights)
    else:
        model = XGBClassifier(random_state = 15, eta = 0.3, colsample_bynode = 0.7)
        model.fit(X_train_original, y_train_original, sample_weight=classes_weights, feature_weights = feature_weights)
    ## accuracy
    # X_test = X_test_original.loc[:, X_test_original.columns != 'SK_ID_CURR']
    y_pred = model.predict(X_test_original)
    predictions = [round(value) for value in y_pred]
    acc = accuracy_score(y_test_original, predictions)* 100.0
    i_dict_acc = {'participant_id':[p_id],'iteration':[iteration],'fs':[fs], 'accuracy':[acc]}
    
    ##add predictions to original test set 
    train_df_test_bin_ = train_df_test_bin.copy()   
    train_df_test_bin_.insert(loc=1, column="Predicted_Result", value = predictions)

    ## fairness metrics
    i_dict_group = group_fairness(sensitive_attrs, train_df_test_bin_)
    i_dict_group['fs'] = [fs]*len(i_dict_group['Feature'])
    i_dict_group['iteration'] = [iteration]*len(i_dict_group['Feature'])
    i_dict_group['participant_id'] = [p_id]*len(i_dict_group['Feature'])
    
    ##
    i_dict_indv={}
    X_without_sensitive_attr = X_test_original.drop(columns = ["CODE_GENDER_LE","AGE",'NAME_FAMILY_STATUS_LE'],inplace = False)
    consistency = consistency_score_(X_without_sensitive_attr, predictions, 10)
    counterfactual_CODE_GENDER = counterfactual(X_test_original.copy(),y_pred,model,"CODE_GENDER_LE")
    counterfactual_AGE = counterfactual(X_test_original.copy(),y_pred,model,"AGE")
    counterfactual_NAME_FAMILY_STATUS = counterfactual(X_test_original.copy(),y_pred,model,"NAME_FAMILY_STATUS_LE")
    i_dict_indv = {'participant_id':[p_id],'iteration':[iteration],'fs':[fs], 'consistency':[consistency], 'counterfactual_CODE_GENDER':[counterfactual_CODE_GENDER], 'counterfactual_AGE':[counterfactual_AGE], 'counterfactual_NAME_FAMILY_STATUS':[counterfactual_NAME_FAMILY_STATUS]}
    return pd.DataFrame(i_dict_acc), pd.DataFrame(i_dict_group), pd.DataFrame(i_dict_indv)

def oneoff_training_evaluation(X_train_original, y_train_original,
                            X_test_original, y_test_original,
                            train_df_test_bin, test_df,
                            sensitive_attrs, fs, feedback_df,
                           onlyUnfair, useFeatureWeights, path, method_indicative_fileName_exte):
    """
        onlyUnfair: Boolean if True take only 'unfair' labelled instances
        method_indicative_fileName_exte: Str
    """
    accuracy = []
    group_fairness_metrics = []
    indiv_fairness_metrics = []
                               
    ## BEFORE INTEGRATING ANY FEEDBACK
    X_train = X_train_original.copy()
    y_train = y_train_original.copy()
    X_test = X_test_original.copy()
    y_test = y_test_original.copy()
    X_test_bin = train_df_test_bin.copy()
    acc_row, group_row, indv_row = train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                                                             X_test_bin, [],
                                                            sensitive_attrs,
                                                            0, fs, 0, None)
    accuracy.append(acc_row)
    group_fairness_metrics.append(group_row)
    indiv_fairness_metrics.append(indv_row)
                               
    ## INTEGRATE FEEDBACK
    count = 0
    feature_weights = np.array([])
    for j, p_id in enumerate(feedback_df['ID'].unique()):
        print('participant',j)
        feedbackInstances_j = feedback_df[feedback_df['ID']== p_id].sort_values(by=['timestamp'])            
        for k, idx in enumerate(feedbackInstances_j.index):
            ## get application data
            app_id = feedback_df['App ID'].loc[idx]        
            test_df_app_id = test_df[test_df['SK_ID_CURR'] == app_id].loc[:,test_df.columns!='SK_ID_CURR']
            ## get predicted label
            pred_label = feedback_df['PredictedDecision'].loc[idx]
            if pred_label == 'Accepted':
                pred_label = 1
            elif pred_label == 'Rejected':
                pred_label = 0
            else:
                pred_label = None
            ## flip label if user indicated 'unfair'
            feed_label = feedback_df['Attribute'].loc[idx] 
            if feed_label == 'unfair':
                if pred_label == 1:
                    pred_label = 0
                elif pred_label == 0:
                    pred_label = 1
            elif onlyUnfair and (feed_label == 'checked'):
                continue    
            if useFeatureWeights:
                if len(feedback_df['Value'].loc[idx]) == 0:
                    continue
                if len(feature_weights):
                    feature_weights = feature_weights+np.array(feedback_df['Value'].loc[idx])
                else:
                    feature_weights = np.array(feedback_df['Value'].loc[idx])                    
            ##
            count = count + 1
            X_train = pd.concat([X_train, test_df_app_id], ignore_index=True)
            y_train = pd.concat([y_train, pd.DataFrame({'TARGET':[pred_label]})], ignore_index=True)
    ## retrain after integrating all feedback
    if len(feature_weights)>0:
        feature_weights = feature_weights/count
        feature_weights = feature_weights/sum(feature_weights)
    acc_row, group_row, indv_row = train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                                                            X_test_bin, feature_weights,
                                                            sensitive_attrs,
                                                            count, fs, count, None)
    accuracy.append(acc_row)
    group_fairness_metrics.append(group_row)
    indiv_fairness_metrics.append(indv_row)
    ## prepare results for saving
    df_group = None
    df_indiv = None
    df_acc = None
    for i, i_df in enumerate(group_fairness_metrics):
        ## group
        if df_group is not None:
            df_group = pd.concat([df_group, i_df], ignore_index=True)
        else:
            df_group = i_df
        ## individual
        i_df_indv = indiv_fairness_metrics[i]
        if df_indiv is not None:
            df_indiv = pd.concat([df_indiv, i_df_indv], ignore_index=True)
        else:
            df_indiv = i_df_indv
        ## acc
        i_df_acc = accuracy[i]
        if df_acc is not None:
            df_acc = pd.concat([df_acc, i_df_acc], ignore_index=True)
        else:
            df_acc = i_df_acc
    ## save as csv
    df_group.to_csv(path+"group_fairness_"+method_indicative_fileName_exte+".csv", index=False)
    df_indiv.to_csv(path+"individual_fairness_"+method_indicative_fileName_exte+".csv", index=False)
    df_acc.to_csv(path+"accuracy_"+method_indicative_fileName_exte+".csv", index=False)
                               
def iml_training_evaluation(X_train_original, y_train_original,
                            X_test_original, y_test_original, 
                            train_df_test_bin, test_df,
                            sensitive_attrs, fs, feedback_df,
                           onlyUnfair, useFeatureWeights, folder, method_indicative_fileName_exte):
    """
        onlyUnfair: Boolean if True take only 'unfair' labelled instances
        useFeatureWeights: Boolean if True set feature weights from feedback
        method_indicative_fileName_exte: Str
    """
    accuracy = []
    group_fairness_metrics = []
    indiv_fairness_metrics = []
                               
    ## BEFORE INTEGRATING FEEDBACK
    X_train = X_train_original.copy()
    y_train = y_train_original.copy()
    X_test = X_test_original.copy()
    y_test = y_test_original.copy()
    X_test_bin = train_df_test_bin.copy()
    acc_row, group_row, indv_row = train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                                                             X_test_bin, [],
                                                            sensitive_attrs,
                                                            0, fs, 0, None)
    accuracy.append(acc_row)
    group_fairness_metrics.append(group_row)
    indiv_fairness_metrics.append(indv_row)
                               
    ## INTEGRATE FEEDBACK 
    for j, p_id in enumerate(feedback_df['ID'].unique()):
        print('participant',j)
        X_train = X_train_original.copy()
        y_train = y_train_original.copy()
        X_test = X_test_original.copy()
        y_test = y_test_original.copy()
        X_test_bin = train_df_test_bin.copy()
        feedbackInstances_j = feedback_df[feedback_df['ID']== p_id].sort_values(by=['timestamp'])
        count = 0
        for k, idx in enumerate(feedbackInstances_j.index):
            ## get application data
            app_id = feedback_df['App ID'].loc[idx]        
            test_df_app_id = test_df[test_df['SK_ID_CURR'] == app_id].loc[:,test_df.columns!='SK_ID_CURR']
            ## get predicted label
            pred_label = feedback_df['PredictedDecision'].loc[idx]
            # print(pred_label)
            # pred_label = applications_df[applications_df["Application_id"] == app_id]["Predicted_decision"].tolist()[0]
            if pred_label == 'Accepted':
                pred_label = 1
            elif pred_label == 'Rejected':
                pred_label = 0
            else:
                pred_label = None
            ## flip label if user indicated 'unfair'
            feed_label = feedback_df['Attribute'].loc[idx] 
            if feed_label == 'unfair':
                if pred_label == 1:
                    pred_label = 0
                elif pred_label == 0:
                    pred_label = 1
            elif onlyUnfair and (feed_label == 'checked'):
                continue
            feature_weights = []
            if useFeatureWeights:
                feature_weights = feedback_df['Value'].loc[idx]
                if feature_weights == []:
                    continue
            ##
            count = count + 1
            X_train = pd.concat([X_train, test_df_app_id], ignore_index=True)
            y_train = pd.concat([y_train, pd.DataFrame({'TARGET':[pred_label]})], ignore_index=True)
            acc_row, group_row, indv_row = train_and_evaluate_model(X_train, y_train, X_test, y_test, 
                                                                    X_test_bin, feature_weights,
                                                                    sensitive_attrs,
                                                                    count, fs, count-1, p_id)
            # print("iteration: ",count,"p_id: ",p_id,"time: ", timer()-start)
            accuracy.append(acc_row)
            group_fairness_metrics.append(group_row)
            indiv_fairness_metrics.append(indv_row)
    ## prepare results for saving
    df_group = None
    df_indiv = None
    df_acc = None
    for i, i_df in enumerate(group_fairness_metrics):
        ## group
        if df_group is not None:
            df_group = pd.concat([df_group, i_df], ignore_index=True)
        else:
            df_group = i_df
        ## individual
        i_df_indv = indiv_fairness_metrics[i]
        if df_indiv is not None:
            df_indiv = pd.concat([df_indiv, i_df_indv], ignore_index=True)
        else:
            df_indiv = i_df_indv
        ## acc
        i_df_acc = accuracy[i]
        if df_acc is not None:
            df_acc = pd.concat([df_acc, i_df_acc], ignore_index=True)
        else:
            df_acc = i_df_acc
    ## save as csv
    df_group.to_csv(folder+"group_fairness_"+method_indicative_fileName_exte+".csv", index=False)
    df_indiv.to_csv(folder+"individual_fairness_"+method_indicative_fileName_exte+".csv", index=False)
    df_acc.to_csv(folder+"accuracy_"+method_indicative_fileName_exte+".csv", index=False)
    ## add cumulative moving average lines data
    add_cma_data(df_group, ['DemographicParityRatio','AverageOddsDifference'], df_indiv, ['consistency',
                                                                                      'counterfactual_CODE_GENDER',
                                                                                      'counterfactual_AGE',
                                                                                      'counterfactual_NAME_FAMILY_STATUS'], 
                 df_acc, sensitive_attrs, fs)
    df_group.to_csv(folder+"group_fairness_"+method_indicative_fileName_exte+"_with_cma.csv", index=False)
    df_indiv.to_csv(folder+"individual_fairness_"+method_indicative_fileName_exte+"_with_cma.csv", index=False)
    df_acc.to_csv(folder+"accuracy_"+method_indicative_fileName_exte+"_with_cma.csv", index=False)