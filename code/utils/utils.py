import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

from sklearn.utils import class_weight

attributes_names_mapping = dict(zip(['CNT_CHILDREN', 
                                         'AMT_INCOME_TOTAL', 
                                         'AMT_CREDIT', 
                                         'AMT_ANNUITY',
       'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'AGE',
       'YEARS_EMPLOYED', 'YEARS_REGISTRATION', 'YEARS_ID_PUBLISH',
       'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'YEARS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE'],['Number of children', 'Income', 'Loan Credit amount ', 'Loan annuity',
       'Goods Price', "Region's Normalized Population", 'Age',
       'Years in current employment', 'Years since changing registration', 'Years since changing  identity document',
       'Has Mobile', 'Has Employee Phone', 'Has Work Phone', 'Mobile reachable',
       'Has Phone', 'Has email', 'Number of family members', "Region's Rating",
       'Region & City Rating', 'Application Hour',
       'Contact address located in Registration region', 'Work address located in Registration region',
       'Contact address located in work region', 'Contact city in Registration city',
       'Work address in Registration city', 'Contact city in work city',
       'Number of times social circle at default risk for 30 days', 'Number of times social circle defaulted on loan for 30 days',
       'Number of times social circle were at risk to default on loan for 60 days', 'Number of times social circle defaulted on loan for 60 days',
       'Years since changing phone', 'Number of Credit Bureau enquiries 1 hour before application',
       'Number of Credit Bureau enquiries 1 day before application', 'Number of Credit Bureau enquiries 1 week before application',
       'Number of Credit Bureau enquiries 1 month before application', 'Number of Credit Bureau enquiries 3 month before application',
       'Number of Credit Bureau enquiries 1 year before application', 'Installments', 'Gender',
       'Owns Car', 'Owns Property', 'Accompanied while applying',
       'Income type', 'Highest education level',
       'Family status', 'Housing situation', 'Occupation Type',
       'Application Day', 'Employer organization']))

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
        
def manipulate_categ_values(df):
    ## Some categorical data manipulation
    df.replace({'NAME_CONTRACT_TYPE' : { 'Cash loans' :'Fixed', 'Revolving loans' : 'Not Fixed'}},inplace=True)
    df.replace({'FLAG_OWN_CAR' : { "N" :'No', "Y" : 'Yes'}},inplace=True)
    df.replace({'FLAG_OWN_REALTY' : { "N" :'No', "Y" : 'Yes'}},inplace=True)
    df.replace({'FLAG_MOBIL' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_EMP_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_WORK_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_CONT_MOBILE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_EMAIL' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'REG_REGION_NOT_LIVE_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_REGION_NOT_WORK_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'LIVE_REGION_NOT_WORK_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_CITY_NOT_LIVE_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_CITY_NOT_WORK_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'LIVE_CITY_NOT_WORK_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'CODE_GENDER' : { "F" :'Female', "M" :'Male'}},inplace=True)
    df.replace({'YEARS_EMPLOYED' : { -1001 :np.nan}}, inplace=True)

    df['NAME_CONTRACT_TYPE'].fillna("Unknown", inplace = True)
    df['CODE_GENDER'].fillna("Unknown", inplace = True)
    df['FLAG_OWN_CAR'].fillna("Unknown", inplace = True)
    df['FLAG_OWN_REALTY'].fillna("Unknown", inplace = True)
    df['NAME_TYPE_SUITE'].fillna("Unknown", inplace = True)
    df['NAME_INCOME_TYPE'].fillna("Unknown", inplace = True)
    df['NAME_EDUCATION_TYPE'].fillna("Unknown", inplace = True)
    df['NAME_FAMILY_STATUS'].fillna("Unknown", inplace = True)
    df['NAME_HOUSING_TYPE'].fillna("Unknown", inplace = True)
    df['FLAG_MOBIL'].fillna("Unknown", inplace = True)
    df['FLAG_EMP_PHONE'].fillna("Unknown", inplace = True)
    df['FLAG_WORK_PHONE'].fillna("Unknown", inplace = True)
    df['FLAG_CONT_MOBILE'].fillna("Unknown", inplace = True)
    df['FLAG_PHONE'].fillna("Unknown", inplace = True)
    df['FLAG_EMAIL'].fillna("Unknown", inplace = True)
    df['OCCUPATION_TYPE'].fillna("Unknown", inplace = True)
    df['WEEKDAY_APPR_PROCESS_START'].fillna("Unknown", inplace = True)
    df['REG_REGION_NOT_LIVE_REGION'].fillna("Unknown", inplace = True)
    df['REG_REGION_NOT_WORK_REGION'].fillna("Unknown", inplace = True)
    df['LIVE_REGION_NOT_WORK_REGION'].fillna("Unknown", inplace = True)
    df['REG_CITY_NOT_LIVE_CITY'].fillna("Unknown", inplace = True)
    df['REG_CITY_NOT_WORK_CITY'].fillna("Unknown", inplace = True)
    df['LIVE_CITY_NOT_WORK_CITY'].fillna("Unknown", inplace = True)
    df['ORGANIZATION_TYPE'].fillna("Unknown", inplace = True)
    
    
def binning(train_df_test_or, train_df_test):
    """
    train_df_test_or: pandas.DataFrame with data before hot encoding and missing values imputation
    train_df_test:    pandas.DataFrame with data after hot encoding and missing values imputation
    """
    ## binning
    train_df_test_bin = train_df_test_or.copy()
    ## AGE
    for j in train_df_test_bin['AGE'].unique().tolist():
        if j<=25.:
            train_df_test_bin['AGE'].replace([j],"Young Adults (18-25)",inplace=True)
        elif j>25. and j<=40.:
            train_df_test_bin['AGE'].replace([j],"Adults (26-40)",inplace=True)
        elif j>40. and j<=60.:
            train_df_test_bin['AGE'].replace([j],"Middle Age Adults (41-60)",inplace=True)
        elif j>60.:
            train_df_test_bin['AGE'].replace([j],"Older Adults (60+)",inplace=True)
        else:
            train_df_test_bin['AGE'].replace([j],"Unknown",inplace=True)
    
    ## OBS_30_CNT_SOCIAL_CIRCLE
    # train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'] = train_df_test['OBS_30_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<2:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],"0-1",inplace=True)
        elif j<=5:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],"2-5",inplace=True)
        elif j>5:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],">5",inplace=True)
        else:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],"Unknown",inplace=True)

    ## DEF_30_CNT_SOCIAL_CIRCLE
    # train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'] = train_df_test['DEF_30_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<=1:
            train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].replace([j],"0-1",inplace=True)
        elif j>1:
            train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].replace([j],">1",inplace=True)
        else:
            train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].replace([j],"Unknown",inplace=True)

    ## OBS_60_CNT_SOCIAL_CIRCLE
    # train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'] = train_df_test['OBS_60_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<2:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],"0-1",inplace=True)
        elif j<=5:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],"2-5",inplace=True)
        elif j>5:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],">5",inplace=True)
        else:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],"Unknown",inplace=True)

    ## DEF_60_CNT_SOCIAL_CIRCLE
    # train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'] = train_df_test['DEF_60_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j==0:
            train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].replace([j],"0",inplace=True)
        elif j>0:
            train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].replace([j],">0",inplace=True)
        else:
            train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].replace([j],"Unknown",inplace=True)

    ## CNT_CHILDREN
    # train_df_test_bin['CNT_CHILDREN'] = train_df_test['CNT_CHILDREN'].tolist()
    for j in train_df_test_bin['CNT_CHILDREN'].unique().tolist():
        if j<=2:
            train_df_test_bin['CNT_CHILDREN'].replace([j],"0-2",inplace=True)
        elif j>2 and j<=5:
            train_df_test_bin['CNT_CHILDREN'].replace([j],"3-5",inplace=True)
        elif j>5 and j<=9:
            train_df_test_bin['CNT_CHILDREN'].replace([j],"6-9",inplace=True)
        else:
            train_df_test_bin['CNT_CHILDREN'].replace([j],"Unknown",inplace=True)
    
    ## AMT_INCOME_TOTAL
    # train_df_test_bin['AMT_INCOME_TOTAL'] = train_df_test['AMT_INCOME_TOTAL'].tolist()
    for j in train_df_test_bin['AMT_INCOME_TOTAL'].unique().tolist():
        if j<100000:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],"<100K",inplace=True)
        elif j>=100000 and j<=150000:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],"100K-150K",inplace=True)
        elif j>=150000 and j<=200000:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],"150K-200K",inplace=True)
        elif j>=200000 and j<=250000:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],"200K-250K",inplace=True)
        elif j>250000:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],">250K",inplace=True)
        else:
            train_df_test_bin['AMT_INCOME_TOTAL'].replace([j],"Unknown",inplace=True)

    ## AMT_CREDIT
    # train_df_test_bin['AMT_CREDIT'] = train_df_test['AMT_CREDIT'].tolist()
    for j in train_df_test_bin['AMT_CREDIT'].unique().tolist():
        if j<250000:
            train_df_test_bin['AMT_CREDIT'].replace([j],"<250K",inplace=True)
        elif j>=250000 and j<=500000:
            train_df_test_bin['AMT_CREDIT'].replace([j],"250K-500K",inplace=True)
        elif j>=500000 and j<=750000:
            train_df_test_bin['AMT_CREDIT'].replace([j],"500K-750K",inplace=True)
        elif j>=750000 and j<=1000000:
            train_df_test_bin['AMT_CREDIT'].replace([j],"750K-1M",inplace=True)
        elif j>1000000:
            train_df_test_bin['AMT_CREDIT'].replace([j],">1M",inplace=True)
        else:
            train_df_test_bin['AMT_CREDIT'].replace([j],"Unknown",inplace=True)
    
    ## AMT_ANNUITY
    # train_df_test_bin['AMT_ANNUITY'] = train_df_test['AMT_ANNUITY'].tolist()
    for j in train_df_test_bin['AMT_ANNUITY'].unique().tolist():
        if j<10000:
            train_df_test_bin['AMT_ANNUITY'].replace([j],"<10K",inplace=True)
        elif j>=10000 and j<=25000:
            train_df_test_bin['AMT_ANNUITY'].replace([j],"10K-25K",inplace=True)
        elif j>=25000 and j<=50000:
            train_df_test_bin['AMT_ANNUITY'].replace([j],"25K-50K",inplace=True)
        elif j>50000:
            train_df_test_bin['AMT_ANNUITY'].replace([j],">50K",inplace=True)
        else:
            train_df_test_bin['AMT_ANNUITY'].replace([j],"Unknown",inplace=True)

    ## AMT_GOODS_PRICE
    # train_df_test_bin['AMT_GOODS_PRICE'] = train_df_test['AMT_GOODS_PRICE'].tolist()
    for j in train_df_test_bin['AMT_GOODS_PRICE'].unique().tolist():
        if j<100000:
            train_df_test_bin['AMT_GOODS_PRICE'].replace([j],"<100K",inplace=True)
        elif j>=100000 and j<=500000:
            train_df_test_bin['AMT_GOODS_PRICE'].replace([j],"100K-500K",inplace=True)
        elif j>=500000 and j<=1000000:
            train_df_test_bin['AMT_GOODS_PRICE'].replace([j],"500K-1M",inplace=True)
        elif j>1000000:
            train_df_test_bin['AMT_GOODS_PRICE'].replace([j],">1M",inplace=True)
        else:
            train_df_test_bin['AMT_GOODS_PRICE'].replace([j],"Unknown",inplace=True)

    ## REGION_POPULATION_RELATIVE
    # train_df_test_bin['REGION_POPULATION_RELATIVE'] = train_df_test['REGION_POPULATION_RELATIVE'].tolist()
    for j in train_df_test_bin['REGION_POPULATION_RELATIVE'].unique().tolist():
        if j<0.01:
            train_df_test_bin['REGION_POPULATION_RELATIVE'].replace([j],"<0.01",inplace=True)
        elif j>=0.01 and j<=0.02:
            train_df_test_bin['REGION_POPULATION_RELATIVE'].replace([j],"0.01-0.02",inplace=True)
        elif j>=0.02 and j<=0.03:
            train_df_test_bin['REGION_POPULATION_RELATIVE'].replace([j],"0.02-0.03",inplace=True)
        elif j>0.03:
            train_df_test_bin['REGION_POPULATION_RELATIVE'].replace([j],">0.03",inplace=True)
        else:
            train_df_test_bin['REGION_POPULATION_RELATIVE'].replace([j],"Unknown",inplace=True)

    ## YEARS_EMPLOYED
    # train_df_test_bin['YEARS_EMPLOYED'] = train_df_test['YEARS_EMPLOYED'].tolist()
    for j in train_df_test_bin['YEARS_EMPLOYED'].unique().tolist():
        if j<10:
            train_df_test_bin['YEARS_EMPLOYED'].replace([j],"<10",inplace=True)
        elif j>=10 and j<=20:
            train_df_test_bin['YEARS_EMPLOYED'].replace([j],"10-20",inplace=True)
        elif j>=20 and j<=30:
            train_df_test_bin['YEARS_EMPLOYED'].replace([j],"20-30",inplace=True)
        elif j>30:
            train_df_test_bin['YEARS_EMPLOYED'].replace([j],">30",inplace=True)
        else:
            train_df_test_bin['YEARS_EMPLOYED'].replace([j],"Unknown",inplace=True)

    ## YEARS_REGISTRATION
    # train_df_test_bin['YEARS_REGISTRATION'] = train_df_test['YEARS_REGISTRATION'].tolist()
    for j in train_df_test_bin['YEARS_REGISTRATION'].unique().tolist():
        if j<10:
            train_df_test_bin['YEARS_REGISTRATION'].replace([j],"<10",inplace=True)
        elif j>=10 and j<=20:
            train_df_test_bin['YEARS_REGISTRATION'].replace([j],"10-20",inplace=True)
        elif j>=20 and j<=30:
            train_df_test_bin['YEARS_REGISTRATION'].replace([j],"20-30",inplace=True)
        elif j>30:
            train_df_test_bin['YEARS_REGISTRATION'].replace([j],">30",inplace=True)
        else:
            train_df_test_bin['YEARS_REGISTRATION'].replace([j],"Unknown",inplace=True)

    ## YEARS_ID_PUBLISH
    # train_df_test_bin['YEARS_ID_PUBLISH'] = train_df_test['YEARS_ID_PUBLISH'].tolist()
    for j in train_df_test_bin['YEARS_ID_PUBLISH'].unique().tolist():
        if j<5:
            train_df_test_bin['YEARS_ID_PUBLISH'].replace([j],"<5",inplace=True)
        elif j>=5 and j<=10:
            train_df_test_bin['YEARS_ID_PUBLISH'].replace([j],"5-10",inplace=True)
        elif j>10:
            train_df_test_bin['YEARS_ID_PUBLISH'].replace([j],">10",inplace=True)
        else:
            train_df_test_bin['YEARS_ID_PUBLISH'].replace([j],"Unknown",inplace=True)

    ## CNT_FAM_MEMBERS
    # train_df_test_bin['CNT_FAM_MEMBERS'] = train_df_test['CNT_FAM_MEMBERS'].tolist()
    for j in train_df_test_bin['CNT_FAM_MEMBERS'].unique().tolist():
        if j<=2:
            train_df_test_bin['CNT_FAM_MEMBERS'].replace([j],"<=2",inplace=True)
        elif j>=3 and j<=4:
            train_df_test_bin['CNT_FAM_MEMBERS'].replace([j],"3-4",inplace=True)
        elif j>4:
            train_df_test_bin['CNT_FAM_MEMBERS'].replace([j],">5",inplace=True)
        else:
            train_df_test_bin['CNT_FAM_MEMBERS'].replace([j],"Unknown",inplace=True)

    ## REGION_RATING_CLIENT
    # train_df_test_bin['REGION_RATING_CLIENT'] = train_df_test['REGION_RATING_CLIENT'].tolist()
    for j in train_df_test_bin['REGION_RATING_CLIENT'].unique().tolist():
        if j==1:
            train_df_test_bin['REGION_RATING_CLIENT'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['REGION_RATING_CLIENT'].replace([j],"2",inplace=True)
        elif j==3:
            train_df_test_bin['REGION_RATING_CLIENT'].replace([j],"3",inplace=True)
        else:
            train_df_test_bin['REGION_RATING_CLIENT'].replace([j],"Unknown",inplace=True)

    ## REGION_RATING_CLIENT_W_CITY
    # train_df_test_bin['REGION_RATING_CLIENT_W_CITY'] = train_df_test['REGION_RATING_CLIENT_W_CITY'].tolist()
    for j in train_df_test_bin['REGION_RATING_CLIENT_W_CITY'].unique().tolist():
        if j==1:
            train_df_test_bin['REGION_RATING_CLIENT_W_CITY'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['REGION_RATING_CLIENT_W_CITY'].replace([j],"2",inplace=True)
        elif j==3:
            train_df_test_bin['REGION_RATING_CLIENT_W_CITY'].replace([j],"3",inplace=True)
        else:
            train_df_test_bin['REGION_RATING_CLIENT_W_CITY'].replace([j],"Unknown",inplace=True)

    ## HOUR_APPR_PROCESS_START
    # train_df_test_bin['HOUR_APPR_PROCESS_START'] = train_df_test['HOUR_APPR_PROCESS_START'].tolist()
    for j in train_df_test_bin['HOUR_APPR_PROCESS_START'].unique().tolist():
        if j<=9:
            train_df_test_bin['HOUR_APPR_PROCESS_START'].replace([j],"0-9",inplace=True)
        elif j>9 and j<=13:
            train_df_test_bin['HOUR_APPR_PROCESS_START'].replace([j],"9-13",inplace=True)
        elif j>13 and j<=17:
            train_df_test_bin['HOUR_APPR_PROCESS_START'].replace([j],"13-17",inplace=True)
        elif j>17:
            train_df_test_bin['HOUR_APPR_PROCESS_START'].replace([j],">17",inplace=True)
        else:
            train_df_test_bin['HOUR_APPR_PROCESS_START'].replace([j],"Unknown",inplace=True)

    ## YEARS_LAST_PHONE_CHANGE
    # train_df_test_bin['YEARS_LAST_PHONE_CHANGE'] = train_df_test['YEARS_LAST_PHONE_CHANGE'].tolist()
    for j in train_df_test_bin['YEARS_LAST_PHONE_CHANGE'].unique().tolist():
        if j<=2:
            train_df_test_bin['YEARS_LAST_PHONE_CHANGE'].replace([j],"<=2",inplace=True)
        elif j>2 and j<=5:
            train_df_test_bin['YEARS_LAST_PHONE_CHANGE'].replace([j],"3-5",inplace=True)
        elif j>5:
            train_df_test_bin['YEARS_LAST_PHONE_CHANGE'].replace([j],">5",inplace=True)
        else:
            train_df_test_bin['YEARS_LAST_PHONE_CHANGE'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_HOUR
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'] = train_df_test['AMT_REQ_CREDIT_BUREAU_HOUR'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'].unique().tolist():
        if j==0:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'].replace([j],"0",inplace=True)
        elif j==1:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'].replace([j],"2",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_HOUR'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_DAY
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'] = train_df_test['AMT_REQ_CREDIT_BUREAU_DAY'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'].unique().tolist():
        if j==0:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'].replace([j],"0",inplace=True)
        elif j==1:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'].replace([j],"2",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_DAY'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_WEEK
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'] = train_df_test['AMT_REQ_CREDIT_BUREAU_WEEK'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'].unique().tolist():
        if j==0:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'].replace([j],"0",inplace=True)
        elif j==1:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'].replace([j],"2",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_WEEK'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_MON
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'] = train_df_test['AMT_REQ_CREDIT_BUREAU_MON'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'].unique().tolist():
        if j==0:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'].replace([j],"0",inplace=True)
        elif j==1:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'].replace([j],"2",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_MON'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_QRT
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'] = train_df_test['AMT_REQ_CREDIT_BUREAU_QRT'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'].unique().tolist():
        if j==0:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'].replace([j],"0",inplace=True)
        elif j==1:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'].replace([j],"1",inplace=True)
        elif j==2:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'].replace([j],"2",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_QRT'].replace([j],"Unknown",inplace=True)

    ## AMT_REQ_CREDIT_BUREAU_YEAR
    # train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'] = train_df_test['AMT_REQ_CREDIT_BUREAU_YEAR'].tolist()
    for j in train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'].unique().tolist():
        if j<3:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'].replace([j],"<3",inplace=True)
        elif j>=3 and j<=4:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'].replace([j],"3-4",inplace=True)
        elif j>4:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'].replace([j],">4",inplace=True)
        else:
            train_df_test_bin['AMT_REQ_CREDIT_BUREAU_YEAR'].replace([j],"Unknown",inplace=True)
            
    return train_df_test_bin

def add_cma_data(df_group, group_fair, df_indiv, indiv_fair, df_acc, sensitive_attrs, fs):
    ## expand dfs with one CMA column per metric
    for gf in group_fair:
        df_group['CMA_'+gf] = [None]*len(df_group[gf])
    for idf in indiv_fair:
        df_indiv['CMA_'+idf] = [None]*len(df_indiv[idf])
    df_acc['CMA_accuracy'] = [None]*len(df_acc['accuracy'])
    ##
    for p,p_id in enumerate(df_group['participant_id'].unique()):     
        if isinstance(p_id, str):
            ## GROUP FAIRNESS
            for i,sens_attr in enumerate(sensitive_attrs):           
                for j,gf in enumerate(group_fair):
                    # for k,fs_i in enumerate(fs):
                    cma = df_group.loc[(df_group['participant_id']==p_id) & (df_group['Feature']==sens_attr) & (df_group['fs']==fs),[gf]].expanding().mean()[gf].tolist()
                    df_group.loc[(df_group['participant_id']==p_id) & (df_group['Feature']==sens_attr) & (df_group['fs']==fs),['CMA_'+gf]] = cma
            # ## INDIVIDUAL FAIRNESS
            for i,idf in enumerate(indiv_fair):
                cma = df_indiv.loc[(df_indiv['participant_id']==p_id) & (df_indiv['fs']==fs),[idf]].expanding().mean()[idf].tolist()
                df_indiv.loc[(df_indiv['participant_id']==p_id) & (df_indiv['fs']==fs),['CMA_'+idf]] = cma
            ## ACCURACY
            cma = df_acc.loc[(df_acc['participant_id']==p_id) & (df_acc['fs']==fs),['accuracy']].expanding().mean()['accuracy'].tolist()
            df_acc.loc[(df_acc['participant_id']==p_id) & (df_acc['fs']==fs),['CMA_accuracy']] = cma

def get_percentage_change_oneoff(df_group, group_fair, df_indiv, indiv_fair, df_acc, sensitive_attrs, fs):
    perc_change_dict = {} ## percentage change (value-baseline)/baseline*100
    ## GROUP FAIRNESS
    for i,sens_attr in enumerate(sensitive_attrs):          
        for j,gf in enumerate(group_fair):
            ## get baseline value
            df_p_null = df_group[df_group['iteration']==0]  
            df = df_p_null[df_p_null['Feature']==sens_attr]
            df = df[df['fs']==fs] 
            baseline = df[gf].tolist()[0]
            ## get diff from baseline
            df_i_non0 = df_group[df_group['iteration']!=0]
            df = df_i_non0[df_i_non0['Feature']==sens_attr]
            df = df[df['fs']==fs]
            if baseline:
                perc_change_dict[sens_attr+'_'+gf] = ((df[gf].tolist()[0]-baseline)/abs(baseline))*100
            else:
                perc_change_dict[sens_attr+'_'+gf] = np.inf
    ## INDIVIDUAL FAIRNESS
    for i,idf in enumerate(indiv_fair):
        ## get baseline value
        df = df_indiv[df_indiv['iteration']==0]               
        baseline = df[idf].tolist()[0]
        ## get diff from baseline
        df = df_indiv[df_indiv['iteration']!=0]
        if baseline:
            perc_change_dict[idf] = ((df[idf].tolist()[0]-baseline)/abs(baseline))*100
        else:
            perc_change_dict[idf] = np.inf
    ## ACCURACT
    ## get baseline value
    df_i_0 = df_acc[df_acc['iteration']==0]
    df = df_i_0[df_i_0['fs']==fs]                
    baseline = df['accuracy'].tolist()[0]
    ## get diff from baseline
    df_i_non0 = df_acc[df_acc['iteration']!=0]
    df = df_i_non0[df_i_non0['fs']==fs]
    # av_diff_vec.append(((df['accuracy'].tolist()[0]-baseline)/baseline)*100)
    if baseline:
        perc_change_dict['accuracy'] = ((df['accuracy'].tolist()[0]-baseline)/abs(baseline))*100
    else:
        perc_change_dict['accuracy'] = np.inf
    return pd.DataFrame([perc_change_dict])

def get_percentage_change_IML(df_group, group_fair, df_indiv, indiv_fair, sensitive_attrs, fs):
    perc_change_dict = {} ## percentage change (value-baseline)/baseline*100
    cma_perc_change_dict = {} 
    p_ids = []
    for p,p_id in enumerate(df_group['participant_id'].unique()):     
        if isinstance(p_id, str): 
            p_ids.append(p_id)
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):          
                for j,gf in enumerate(group_fair):
                    diff = []
                    cma_diff = []
                    ## get baseline value
                    df_p_null = df_group[df_group['participant_id'].isnull()]  
                    df = df_p_null[df_p_null['Feature']==sens_attr]
                    df = df[df['fs']==fs] 
                    baseline = df[gf].tolist()[0]
                    ## get diff from baseline
                    df = df_p[df_p['Feature']==sens_attr]
                    df = df[df['fs']==fs]
                    diff.extend([df[gf].tolist()[-1]-baseline])
                    ## get diff of last iteration in CMA from baseline
                    cma_diff.extend([df['CMA_'+gf].tolist()[-1]-baseline])
                    ###
                    if sens_attr+'_'+gf not in perc_change_dict:
                        perc_change_dict[sens_attr+'_'+gf] = []
                        cma_perc_change_dict[sens_attr+'_'+gf] = []
                    if baseline:
                        perc_change_dict[sens_attr+'_'+gf].append((diff[0]/abs(baseline))*100)
                        cma_perc_change_dict[sens_attr+'_'+gf].append((cma_diff[0]/abs(baseline))*100)
                    else:
                        perc_change_dict[sens_attr+'_'+gf].append(np.inf)
                        cma_perc_change_dict[sens_attr+'_'+gf].append(np.inf)
            ## INDIVIDUAL FAIRNESS
            df_p = df_indiv[df_indiv['participant_id']==p_id]
            for i,idf in enumerate(indiv_fair):
                diff = []
                cma_diff = []
                ## get baseline value
                df_p_null = df_indiv[df_indiv['participant_id'].isnull()]
                df = df_p_null[df_p_null['fs']==fs]                
                baseline = df[idf].tolist()[0]
                ## get diff from baseline
                df = df_p[df_p['fs']==fs]
                diff.extend([df[idf].tolist()[-1]-baseline])
                ## get diff of CMA from baseline
                cma_diff.extend([df['CMA_'+idf].tolist()[-1]-baseline])
                ###             
                if idf not in perc_change_dict:
                    perc_change_dict[idf] = []
                    cma_perc_change_dict[idf] = []
                if baseline:
                    perc_change_dict[idf].append((diff[0]/abs(baseline))*100)
                    cma_perc_change_dict[idf].append((cma_diff[0]/abs(baseline))*100)
                else:
                    perc_change_dict[idf].append(np.inf)
                    cma_perc_change_dict[idf].append(np.inf)
    perc_change_dict['participant_id'] = p_ids
    cma_perc_change_dict['participant_id'] = p_ids
    return pd.DataFrame(perc_change_dict), pd.DataFrame(cma_perc_change_dict)
    
def k_means_optimize_parameter(array_of_vectors, parameters, metric):
    """
        array_of_vectors: List of lists containing the vectors 
        parameters: List of number of clusters for grid searching
        metric: Str in {'mean','median'}
    """
    best_score = -1
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameters:
        if metric=='mean':
            kmeans_model = KMeans(n_clusters=p, init='k-means++', random_state=13, n_init='auto')
        # elif metric=='median':
        #     kmeans_model = KMedoids(n_clusters=p, init='k-medoids++', random_state=13)
        kmeans_model.fit(array_of_vectors)          # fit model on dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(array_of_vectors, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores
        # print('Parameter:', p, 'Score', ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    return silhouette_scores, best_score, best_grid

# def k_means_pca(array_of_vectors, n_clusters, metric, n_components=2):
#     """
#         array_of_vectors: List of lists containing the vectors 
#         n_clusters: Number of clusters for k-means
#         n_components: number of components for PCA
#         metric: Str in {'mean','median'}
#     """
#     if metric=='mean':
#         kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=13, n_init='auto')
#     # elif metric=='median':
#     #     kmeans_model = KMedoids(n_clusters=n_clusters, init='k-medoids++', random_state=13)
#     kmeans_model.fit(array_of_vectors)
#     cluster_ids = kmeans_model.labels_
#     cluster_centroids = kmeans_model.cluster_centers_
#     ##
#     pca = PCA(n_components=2)
#     array_of_vectors_standard = StandardScaler().fit_transform(array_of_vectors)
#     pca.fit(array_of_vectors_standard)
#     array_of_vectors_pca = pca.transform(array_of_vectors_standard)
#     return cluster_ids, cluster_centroids, array_of_vectors_pca