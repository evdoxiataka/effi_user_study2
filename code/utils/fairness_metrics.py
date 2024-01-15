from fairlearn.metrics import MetricFrame, selection_rate
from aif360.sklearn.metrics import consistency_score
from utils.utils import attributes_names_mapping
import numpy as np
import pandas as pd

attribute_bins_mapping = {'AMT_INCOME_TOTAL':['<100K','100K-150K','150K-200K','200K-250K','>250K'],
                          'AMT_CREDIT':["<250K","250K-500K","500K-750K","750K-1M",">1M"],
                         'AMT_ANNUITY':["<10K","10K-25K","25K-50K",">50K"],
                         'AMT_GOODS_PRICE':["<100K","100K-500K","500K-1M",">1M"],
                         'AMT_REQ_CREDIT_BUREAU_HOUR':["0","1","2"],
                         'AMT_REQ_CREDIT_BUREAU_DAY':["0","1","2"],
                         'AMT_REQ_CREDIT_BUREAU_WEEK':["0","1","2"],
                         'AMT_REQ_CREDIT_BUREAU_MON':["0","1","2"],
                         'AMT_REQ_CREDIT_BUREAU_QRT':["0","1","2"],
                          'AMT_REQ_CREDIT_BUREAU_YEAR':["<3","3-4",">4"],
                         'FLAG_MOBIL':['Yes','No'],
                         'FLAG_CONT_MOBILE':['Yes','No'],
                         'REG_REGION_NOT_LIVE_REGION':['Same','Different'],
                         'DEF_60_CNT_SOCIAL_CIRCLE':["0",">0"]}



def DPR_(df, sensitive_feature):
    """
        df: pandas.DataFrame with TARGET col             : y_true
                                  Predicted_Result col   : y_pred
                                  <sensitive_feature> col: sensitive_features
    """
    mf = MetricFrame(metrics = selection_rate, 
                    y_true = df["TARGET"], 
                    y_pred = df["Predicted_Result"], 
                    sensitive_features = df[sensitive_feature])
    groups_dict = {}
    for group in list(mf.by_group.keys()):
        groups_dict[group] = mf.by_group.loc[group]
        #print(sensitive_feature, "fairness ",min(groups_dict.values()),max(groups_dict.values()))    
    if 'Unknown' in groups_dict:
        groups_dict.pop('Unknown')
    dp = None
    if len(groups_dict) > 1:  
        dp = min(groups_dict.values()) / max(groups_dict.values())
    return dp
    
def demographic_parity_ratio_(df, sensitive_feature):
    """
        df: pandas.DataFrame with TARGET col             : y_true
                                  Predicted_Result col   : y_pred
                                  <sensitive_feature> col: sensitive_features
    """
    mf = MetricFrame(metrics = selection_rate, 
                    y_true = df["TARGET"], 
                    y_pred = df["Predicted_Result"], 
                    sensitive_features = df[sensitive_feature])
    groups_dict = mf.by_group.to_dict()
    if 'Unknown' in groups_dict:
        groups_dict.pop('Unknown')
    if len(groups_dict) == 1:
        # dpr = 'Not defined'
        # max_sel_rate = ("",'Second value \n not available')
        dpr = 1. ## not defined - > set to fairest value
        for group in attribute_bins_mapping[sensitive_feature]:
            if group!=list(groups_dict.keys())[0]:
                max_sel_rate = (group,None)
                break
        min_sel_rate = [list(groups_dict.keys())[0],round(list(groups_dict.values())[0],2)]
    else:
        min_sel_rate = list(min(groups_dict.items(), key=lambda x: x[1])) ## tuple (group,sel_rate)
        max_sel_rate = list(max(groups_dict.items(), key=lambda x: x[1])) 
        dpr = round(min(groups_dict.values()) / max(groups_dict.values()),2)
        max_sel_rate[1] = round(max_sel_rate[1],2)
        min_sel_rate[1] = round(min_sel_rate[1],2)
    ## results
    res = []
    res.append({'group':str(min_sel_rate[0]),'total':None,'accepted':None,'acceptance_rate':min_sel_rate[1]})
    res.append({'group':str(max_sel_rate[0]),'total':None,'accepted':None,'acceptance_rate':max_sel_rate[1]})
    res.append({'group':'DPR','total':None,'accepted':None,'acceptance_rate':dpr})    
    return res

def AOD_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    if 'Unknown' in groups:
        groups.remove('Unknown')
    groups_TPR_dict = {}
    groups_FPR_dict = {}
    for group in groups:
        target_value_counts = df[df[sensitive_feature] == group]["TARGET"].value_counts()
        ## TPR
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])        
        if 1 in target_value_counts:
            P = int(target_value_counts[1])
        else:
            P = 0  
        if P:
            groups_TPR_dict[group] = TP/P
        else:
            groups_TPR_dict[group] = None
        ## FPR
        FP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                    (df["TARGET"] == 0) & 
                                   (df[sensitive_feature] == group)])
        if 0 in target_value_counts:
            N = int(target_value_counts[0])
        else:
            N = 0
        if N:
            groups_FPR_dict[group] = FP/N
        else:
            groups_FPR_dict[group] = None
    if len(groups) == 1:
        second_group = ''
        for group in attribute_bins_mapping[sensitive_feature]:
            if group!=list(groups_TPR_dict.keys())[0]:
                second_group = group
                break
        ## TPR
        # min_TPR = ("",'Second value \n not available')
        min_TPR = (second_group+', TP',None)
        if list(groups_TPR_dict.values())[0] is not None:
            max_TPR = [list(groups_TPR_dict.keys())[0]+', TP',list(groups_TPR_dict.values())[0]]
        else:
            # max_TPR = (list(groups_TPR_dict.keys())[0]+', TP','None actual accepted for a second value available')
            max_TPR = (list(groups_TPR_dict.keys())[0]+', TP',None)
        ## FPR
        # min_FPR = ("",'Second value \n not available')
        min_FPR = (second_group+', FP',None)
        if list(groups_FPR_dict.values())[0] is not None:
            max_FPR = [list(groups_FPR_dict.keys())[0]+', FP',list(groups_FPR_dict.values())[0]]
        else:
            # max_FPR = (list(groups_FPR_dict.keys())[0]+', FP','None actual rejected for a second value available')
            max_FPR = (list(groups_FPR_dict.keys())[0]+', FP',None)
        # aod = 'Not defined'
        aod = None
    else:    
        ## TPR
        TPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_TPR_dict.items()))
        if len(tmp):
            max_TPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_TPR[0] = str(max_TPR[0])+', \n TP'
            max_TPR[1] = max_TPR[1]
            if len(tmp) ==1:
                first_group = tmp[0][0]
                for group in attribute_bins_mapping[sensitive_feature]:
                    if group!=first_group:
                        # max_sel_rate = (attr,None)
                        min_TPR = (group+', \n TP',None)
                        break
                # min_TPR = ('','None actual accepted for a second value available')
            else:            
                min_TPR = list(min(tmp, key=lambda x: x[1]))
                min_TPR[0] = str(min_TPR[0])+', \n TP'
                min_TPR[1] = min_TPR[1]
                TPR_diff = max_TPR[1] - min_TPR[1]          
        else:
            first_group = attribute_bins_mapping[sensitive_feature][0]
            second_group = attribute_bins_mapping[sensitive_feature][1]
            min_TPR = (first_group+', \n TP',None)
            max_TPR = (second_group+', \n TP',None)
            # min_TPR = ('','None actual accepted for a second value available')
            # max_TPR = ('','None actual accepted for a second value available')            
        ## FPR
        FPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_FPR_dict.items()))
        if len(tmp):
            max_FPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_FPR[0] = str(max_FPR[0])+', \n FP'
            max_FPR[1] = max_FPR[1]
            if len(tmp) ==1:
                first_group = tmp[0][0]
                for group in attribute_bins_mapping[sensitive_feature]:
                    if group!=first_group:
                        # max_sel_rate = (attr,None)
                        min_FPR = (group+', \n FP',None)
                        break
                # min_FPR = ('','None actual rejected for a second value available')
            else:            
                min_FPR = list(min(tmp, key=lambda x: x[1]))
                min_FPR[0] = str(min_FPR[0])+', \n FP'
                min_FPR[1] = min_FPR[1] 
                FPR_diff = max_FPR[1] - min_FPR[1]               
        else:
            first_group = attribute_bins_mapping[sensitive_feature][0]
            second_group = attribute_bins_mapping[sensitive_feature][1]
            min_FPR = (first_group+', \n FP',None)
            max_FPR = (second_group+', \n FP',None)
            # min_FPR = ('','None actual rejected for a second value available')
            # max_FPR = ('','None actual rejected for a second value available')
        ##
        if TPR_diff is not None and FPR_diff is not None:
            aod = (TPR_diff + FPR_diff)/2.
        else:
            # aod = 'Not defined'
            aod = None ## not defined - > set to fairest value
    return aod
    
def average_odds_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    if 'Unknown' in groups:
        groups.remove('Unknown')
    groups_TPR_dict = {}
    groups_FPR_dict = {}
    for group in groups:
        target_value_counts = df[df[sensitive_feature] == group]["TARGET"].value_counts()
        ## TPR
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])        
        if 1 in target_value_counts:
            P = int(target_value_counts[1])
        else:
            P = 0  
        if P:
            groups_TPR_dict[group] = TP/P
        else:
            groups_TPR_dict[group] = None
        ## FPR
        FP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                    (df["TARGET"] == 0) & 
                                   (df[sensitive_feature] == group)])
        if 0 in target_value_counts:
            N = int(target_value_counts[0])
        else:
            N = 0
        if N:
            groups_FPR_dict[group] = FP/N
        else:
            groups_FPR_dict[group] = None
    if len(groups) == 1:
        second_group = ''
        for group in attribute_bins_mapping[sensitive_feature]:
            if group!=list(groups_TPR_dict.keys())[0]:
                second_group = group
                break
        ## TPR
        # min_TPR = ("",'Second value \n not available')
        min_TPR = (second_group+', TP',None)
        if list(groups_TPR_dict.values())[0] is not None:
            max_TPR = [list(groups_TPR_dict.keys())[0]+', TP',round(list(groups_TPR_dict.values())[0],2)]
        else:
            # max_TPR = (list(groups_TPR_dict.keys())[0]+', TP','None actual accepted for a second value available')
            max_TPR = (list(groups_TPR_dict.keys())[0]+', TP',None)
        ## FPR
        # min_FPR = ("",'Second value \n not available')
        min_FPR = (second_group+', FP',None)
        if list(groups_FPR_dict.values())[0] is not None:
            max_FPR = [list(groups_FPR_dict.keys())[0]+', FP',round(list(groups_FPR_dict.values())[0],2)]
        else:
            # max_FPR = (list(groups_FPR_dict.keys())[0]+', FP','None actual rejected for a second value available')
            max_FPR = (list(groups_FPR_dict.keys())[0]+', FP',None)
        # aod = 'Not defined'
        aod = 0
    else:    
        ## TPR
        TPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_TPR_dict.items()))
        if len(tmp):
            max_TPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_TPR[0] = str(max_TPR[0])+', \n TP'
            max_TPR[1] = round(max_TPR[1],2)
            if len(tmp) ==1:
                first_group = tmp[0][0]
                for group in attribute_bins_mapping[sensitive_feature]:
                    if group!=first_group:
                        # max_sel_rate = (attr,None)
                        min_TPR = (group+', \n TP',None)
                        break
                # min_TPR = ('','None actual accepted for a second value available')
            else:            
                min_TPR = list(min(tmp, key=lambda x: x[1]))
                min_TPR[0] = str(min_TPR[0])+', \n TP'
                min_TPR[1] = round(min_TPR[1],2) 
                TPR_diff = max_TPR[1] - min_TPR[1]          
        else:
            first_group = attribute_bins_mapping[sensitive_feature][0]
            second_group = attribute_bins_mapping[sensitive_feature][1]
            min_TPR = (first_group+', \n TP',None)
            max_TPR = (second_group+', \n TP',None)
            # min_TPR = ('','None actual accepted for a second value available')
            # max_TPR = ('','None actual accepted for a second value available')            
        ## FPR
        FPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_FPR_dict.items()))
        if len(tmp):
            max_FPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_FPR[0] = str(max_FPR[0])+', \n FP'
            max_FPR[1] = round(max_FPR[1],2)
            if len(tmp) ==1:
                first_group = tmp[0][0]
                for group in attribute_bins_mapping[sensitive_feature]:
                    if group!=first_group:
                        # max_sel_rate = (attr,None)
                        min_FPR = (group+', \n FP',None)
                        break
                # min_FPR = ('','None actual rejected for a second value available')
            else:            
                min_FPR = list(min(tmp, key=lambda x: x[1]))
                min_FPR[0] = str(min_FPR[0])+', \n FP'
                min_FPR[1] = round(min_FPR[1],2)  
                FPR_diff = max_FPR[1] - min_FPR[1]               
        else:
            first_group = attribute_bins_mapping[sensitive_feature][0]
            second_group = attribute_bins_mapping[sensitive_feature][1]
            min_FPR = (first_group+', \n FP',None)
            max_FPR = (second_group+', \n FP',None)
            # min_FPR = ('','None actual rejected for a second value available')
            # max_FPR = ('','None actual rejected for a second value available')
        ##
        if TPR_diff is not None and FPR_diff is not None:
            aod = round((TPR_diff + FPR_diff)/2.,2)
        else:
            # aod = 'Not defined'
            aod = 0 ## not defined - > set to fairest value
    ## results
    res = []    
    res.append({'group':max_TPR[0],'total':None,'accepted':None,'acceptance_rate':max_TPR[1]})
    res.append({'group':min_TPR[0],'total':None,'accepted':None,'acceptance_rate':min_TPR[1]})
    res.append({'group':max_FPR[0],'total':None,'accepted':None,'acceptance_rate':max_FPR[1]})
    res.append({'group':min_FPR[0],'total':None,'accepted':None,'acceptance_rate':min_FPR[1]})    
    res.append({'group':'AOD','total':None,'accepted':None,'acceptance_rate':aod})
    return res

def value_distribution_(df, attribute):
    groups = list(df[attribute].unique())
    res = []
    for group in groups:
        df_g = df[df[attribute] == group]
        acc = len(df_g[df_g['Predicted_Result'] == 1])
        rej = len(df_g[df_g['Predicted_Result'] == 0])
        # acc_rate = round(acc/(acc+rej),2)
        res.append({'group':str(group), 'Accepted':acc, 'Rejected':rej})
    return res

def DPR_AOD_fairness(attributes_names_mapping, test_df_bin):
    """
        attributes_names_mapping:  Dict of attributes to calculate fairness metrics mapped to user friendly descriptions
        test_df_bin: pandas.DataFrame of binned test set 
    """
    attrs = list(attributes_names_mapping.keys())
    fairness_results = []
    for attr in attrs:
        res = {}
        res['attribute'] = attributes_names_mapping[attr]
        ## DEMOGRAPHIC PARITY (DP) RATIO
        res['dp'] = demographic_parity_ratio_(test_df_bin, attr)        
        ## AVERAGE ODDS DIFFERENCE
        res['ao'] = average_odds_difference_(test_df_bin, attr)
        ## VALUE DISTRIBUTION
        res['value_distribution'] = value_distribution_(test_df_bin, attr)
        ##
        fairness_results.append(res)
    return fairness_results

def consistency_score_(X, Y, k):
    return consistency_score(X,Y,k)

def counterfactual(X, y_pred_init, model, sensitive_attribute):
    counterfactual_value = None
    sens_attr_values = X[sensitive_attribute].unique().tolist()
    if len(sens_attr_values)>1:        
        mean_count = []
        for i, idx in enumerate(X.index):
            ## get application data
            feedback_instance_i = X.loc[idx].to_frame().T
            feedback_instances = None
            y_pred_inits = []
            sens_attr_value = feedback_instance_i[sensitive_attribute].tolist()[0]
            count_same_outcome = 0
            for v in sens_attr_values:
                if v!=sens_attr_value:
                    fd_inst_copy = feedback_instance_i.copy()
                    fd_inst_copy[sensitive_attribute] = v
                    if feedback_instances is not None:
                        feedback_instances = pd.concat([feedback_instances, fd_inst_copy], ignore_index=True)
                    else:
                        feedback_instances = fd_inst_copy                    
                    y_pred_inits.append(y_pred_init[i])
                    # y_pred_cur = model.predict(feedback_instance_i)
                    # # print(y_pred_init[i],y_pred_cur[0],y_pred_init[i] == y_pred_cur[0])
                    # if y_pred_init[i] == y_pred_cur[0]:
                    #     count_same_outcome = count_same_outcome + 1            
            y_pred_currs = model.predict(feedback_instances)
            # print(len(y_pred_inits),len(y_pred_currs))
            bool_arr = np.array(y_pred_inits)==np.array(y_pred_currs)
            count_same_outcome = bool_arr.sum()
            # for i,val in enumerate(y_pred_currs):
            #     if y_pred_inits[i]==val:
            #         count_same_outcome = count_same_outcome + 1
            mean_count.append(count_same_outcome/(len(sens_attr_values)-1))
        counterfactual_value = sum(mean_count)/len(mean_count)
    return counterfactual_value
    
def group_fairness(sensitive_attrs, train_df_test_bin):
    """
        sensitive_attrs:      List of sensitive attributes
        train_df_test_binned: pandas.DataFrame of binned test set 
    """
    fairness_metrics_per_feature = {} 
    fairness_metrics_per_feature["Feature"] = []
    fairness_metrics_per_feature["DemographicParityRatio"] = []
    fairness_metrics_per_feature["AverageOddsDifference"] = []
    for attr in sensitive_attrs:
        fairness_metrics_per_feature['Feature'].append(attr)
        # fairness_metrics_per_feature['Feature'].append(attributes_names_mapping[attr])
        ## DEMOGRAPHIC PARITY (DP) RATIO
        dp = DPR_(train_df_test_bin, attr)
        fairness_metrics_per_feature["DemographicParityRatio"].append(dp)
       
        ## AVERAGE ODDS DIFFERENCE
        aod = AOD_(train_df_test_bin, attr)
        fairness_metrics_per_feature['AverageOddsDifference'].append(aod)

    return fairness_metrics_per_feature