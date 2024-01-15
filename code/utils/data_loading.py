import os
import csv
import ast
import pandas as pd
import numpy as np

def get_questionnaires_responses(folderPath, fileEnding):
    participants_responses_dict = {} ## <prolific_id>: Dict <question_id>: response
    ##
    for root,dirs,files in os.walk(folderPath):
        for file in files:
            if file.endswith(fileEnding):
                with open(folderPath+"\\"+file, "r") as f:
                    reader = csv.reader(f)
                    next(reader) 
                    qa_dict = {}
                    for row in reader:
                        part_responses = row[3] 
                        for qa in ast.literal_eval(part_responses):
                            qa_dict[qa['question']] = qa['response']
                f.close()
                user_id = file.replace(fileEnding, "")
                participants_responses_dict[user_id] = qa_dict
    return participants_responses_dict

def get_feedback(folderPath, fileEnding):
    participants_responses_dict = {} ## <prolific_id>: Dict <question_id>: response
    ##
    for root,dirs,files in os.walk(folderPath):
        for file in files:
            if file.endswith(fileEnding):
                with open(folderPath+"\\"+file, "r",encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) 
                    feedback_dict = {}
                    for row in reader:                       
                        if row[5] == 'OKBUTTON_CLICKED_DECIDE_MODAL_Applications_List':
                            weights = row[7]
                            weights = ast.literal_eval(weights)
                            feedback_dict[row[3]]={'label':row[6],'init_weights':weights['initial_weights'],'changed_weights':weights['changed_weights']}

                f.close()
                user_id = file.replace(fileEnding, "")
                #print(file.replace(fileEnding, "") )
                participants_responses_dict[user_id] = feedback_dict
    return participants_responses_dict

def get_training(folderPath, fileEnding):
    participants_responses_dict = {} ## <prolific_id>: Dict <question_id>: response
    ##
    for root,dirs,files in os.walk(folderPath):
        for file in files:
            if file.endswith(fileEnding):
                with open(folderPath+"\\"+file, "r",encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader) 
                    feedback_dict = {}
                    for row in reader:  
                        fairness = row[4]
                        fairness = ast.literal_eval(fairness)
                        predicted_decision = row[5]
                        predicted_decision = ast.literal_eval(predicted_decision)
                        predicted_acceptance_confidence = row[6]
                        predicted_acceptance_confidence = ast.literal_eval(predicted_acceptance_confidence)
                        predictied_decision_indication_change = row[7]
                        predictied_decision_indication_change = ast.literal_eval(predictied_decision_indication_change)
                        predictied_acceptance_conf_indication_change = row[8]
                        predictied_acceptance_conf_indication_change = ast.literal_eval(predictied_acceptance_conf_indication_change)
                        feedback_dict[row[3]]={'fairness':fairness,'predicted_decision':predicted_decision,'predicted_acceptance_confidence':predicted_acceptance_confidence,'predictied_decision_indication_change':predictied_decision_indication_change,'predictied_acceptance_conf_indication_change':predictied_acceptance_conf_indication_change}

                f.close()
                user_id = file.replace(fileEnding, "")
                # print(file.replace(fileEnding, "") )
                participants_responses_dict[user_id] = feedback_dict
    return participants_responses_dict

def get_feedback_df(prolific_export_filePath, interaction_logs_filePath):
    prolific_export = pd.read_excel(prolific_export_filePath)
    ## GET data of only approved participants
    prolific_export = prolific_export[prolific_export['ACTION']=='Approved']
    
    ## GET prolific id of approved participants
    participants = prolific_export['Participant id'].tolist()
    
    ## GET FEEDBACK
    feedback_df = pd.DataFrame()    
    for p_id in participants:    
        file_name = p_id+".csv"    
        file = pd.read_csv(interaction_logs_filePath+file_name, delimiter=',')
        
        ## Conversion of App ID to int     
        file["App ID"] = file["App ID"].fillna(0.0).astype(int)
    
        ## Reset_index 
        file.index = np.arange(0, len(file))
        
        ## Dropping Login
        file = file[file.Function != "Login"]
        file.reset_index(drop = True, inplace = True)    
    
        feedback_df = pd.concat([feedback_df,file],axis=0)
        
    feedback_df.drop(columns = "Pattern",inplace = True)
    feedback_df.reset_index(drop = True, inplace = True)

    ## Dropping the entires where function chosen is apply_refine_search which is filtering for attribute 
    ## but there is no attribute present that was filtered for. 
    
    df_with_null_attribute = feedback_df[feedback_df["Attribute"].isna()]
    null_attribute_index = df_with_null_attribute[df_with_null_attribute["Function"] == "apply_refine_search"].index
    null_attribute_index
    
    print(len(null_attribute_index))
    
    feedback_df.drop(index = null_attribute_index,inplace = True)
    
    feedback_df.reset_index(drop = True, inplace = True)
    return feedback_df