import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs):
    ## get max length of feedback to define lim of x-axis
    feed_len = []
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        df_p = df_group[df_group['participant_id']==p_id]
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                df = df_p[df_p['Feature']==sens_attr]
                df = df[df['fs']==fs]
                feed_len.append(len(df))
    return max(feed_len)
    
def joint_plot_all_participants(title, folder, filename, image_type, sensitive_attrs, group_fair, group_fair_codes, fs, df_group, df_acc, colors, show_timeseries, show_cma):
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    ## 
    fig, axes = plt.subplots(len(sensitive_attrs)+1, len(group_fair), figsize=(50, 45), layout="constrained")
    fig.suptitle(title)

    ## TIME SERIES
    if show_timeseries:
        flag = True
        ## GROUP FAIRNESS
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                df = df_group[df_group['Feature']==sens_attr]
                df = df[df['fs']==fs]
                ## draw one curve per participant
                for p,p_id in enumerate(df['participant_id'].unique()):
                    df_p = df[df['participant_id']==p_id]
                    ## draw time series
                    if isinstance(p_id, str):## not None                    
                        line, = axes[i,j].plot(df_p['iteration'], df_p[gf], c = colors[2])  
                        if flag:
                            line.set_label('Feedback Integration')
                            flag = False        
        ## ACCURACY
        df = df_acc[df_acc['fs']==fs]
        ## draw one curve per participant
        for p,p_id in enumerate(df['participant_id'].unique()):
            df_p = df[df['participant_id']==p_id]
            ## draw time series
            if isinstance(p_id, str):## not None 
                axes[len(sensitive_attrs),0].plot(df_p['iteration'], df_p['accuracy'], c = colors[2])
    ## CUMULATIVE MOVING AVERAGE LINES
    if show_cma:  
        flag = True
        ## GROUP FAIRNESS
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                df = df_group[df_group['Feature']==sens_attr]
                df = df[df['fs']==fs]
                ## draw one curve per participant
                for p,p_id in enumerate(df['participant_id'].unique()):
                    df_p = df[df['participant_id']==p_id]
                    ## draw cma line
                    if isinstance(p_id, str):## not None
                        line, = axes[i,j].plot(df_p['iteration'], df_p['CMA_'+gf], c = colors[0], linewidth=1)
                        if flag:
                            line.set_label('CMA of Feedback Integration')
                            flag = False        
        ## ACCURACY
        df = df_acc[df_acc['fs']==fs]
        ## draw one curve per participant
        for p,p_id in enumerate(df['participant_id'].unique()):
            df_p = df[df['participant_id']==p_id]
            ## ddraw cma line
            if isinstance(p_id, str):## not None 
                axes[len(sensitive_attrs),0].plot(df_p['iteration'], df_p['CMA_accuracy'], c = colors[0], linewidth=1)
    ## BASELINE LINES
    flag = True
    ## GROUP FAIRNESS
    for i,sens_attr in enumerate(sensitive_attrs):    
        for j,gf in enumerate(group_fair):
            df = df_group[df_group['Feature']==sens_attr]
            df = df[df['fs']==fs]
            ## draw baseline
            df_p = df[df['participant_id'].isnull()]                
            line, = axes[i,j].plot([l for l in range(xlim)], [df_p[gf].tolist()[0]]*xlim, c='black')
            if flag:
                line.set_label('Baseline - No Feedback')
                flag = False
            axes[i,j].set_xlabel("Iteration")
            axes[i,j].set_ylabel(group_fair_codes[j])
            axes[i,j].set_title(sens_attr)      
    ## ACCURACY
    df = df_acc[df_acc['fs']==fs]        
    ## draw baseline
    df_p = df[df['participant_id'].isnull()]                
    axes[len(sensitive_attrs),0].plot([l for l in range(xlim)], [df_p['accuracy'].tolist()[0]]*xlim, c = 'black')   
    
    axes[0,0].legend(bbox_to_anchor=(0.0,2.0),loc='upper left') 
    axes[len(sensitive_attrs),0].set_xlabel("Iteration")
    axes[len(sensitive_attrs),0].set_ylabel('Accuracy %')    
    fig.delaxes(axes[len(sensitive_attrs),1])
    
    fig.savefig(folder+filename+image_type, dpi=300)
    plt.show()

def line_graphs_of_participant(title, folder, image_type, sensitive_attrs, group_fair, group_fair_codes, 
                               fs, df_group, df_acc, colors, show_cma, participant_id):
    """
        show_cma: Boolean, if True plot Cumulative Moving Average Lines
    """
    title_code = {'CODE_GENDER':'GENDER','AGE':'AGE','NAME_FAMILY_STATUS':'MARIT. STAT.'}
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        if isinstance(p_id, str) and p_id == participant_id:        
            ##
            fig, axes = plt.subplots(len(sensitive_attrs), len(group_fair), figsize=(25, 25), layout="constrained")
            fig.suptitle(title.format(p_id))
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):    
                for j,gf in enumerate(group_fair):
                    ## draw time series
                    df = df_p[df_p['Feature']==sens_attr]
                    df = df[df['fs']==fs]                   
                    axes[i,j].plot(df['iteration'], df[gf], c = colors[2], linewidth=2, label = 'Feedback Integration') #colors[j]    
                    ## draw cma line
                    if show_cma:
                        axes[i,j].plot(df['iteration'], df['CMA_'+gf], c = colors[0], linewidth=2, label = 'CMA of Feedback Integration')
                    ## draw baseline
                    df_p_null = df_group[df_group['participant_id'].isnull()]  
                    df = df_p_null[df_p_null['Feature']==sens_attr]
                    df = df[df['fs']==fs] 
                    axes[i,j].plot([l for l in range(xlim)], [df[gf].tolist()[0]]*xlim,c='black', label = 'Baseline - No Feedback')             
                    axes[i,j].set_xlabel("Iteration")
                    axes[i,j].set_ylabel(group_fair_codes[j])
                    axes[i,j].set_title(title_code[sens_attr])   
            axes[0,0].legend(bbox_to_anchor=(0.3,1.0),loc='upper left')             
            fig.savefig(folder+'{}'.format(p_id)+image_type, dpi=300)
            plt.show()
            
def plots_per_participant(title, folder, image_type, sensitive_attrs, group_fair, group_fair_codes, fs, df_group, df_acc, colors, show_cma):
    """
        show_cma: Boolean, if True plot Cumulative Moving Average Lines
    """
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        if isinstance(p_id, str):        
            ##
            fig, axes = plt.subplots(len(sensitive_attrs)+1, len(group_fair), figsize=(60, 45), layout="constrained")
            fig.suptitle(title.format(p_id))
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):    
                for j,gf in enumerate(group_fair):
                    ## draw time series
                    df = df_p[df_p['Feature']==sens_attr]
                    df = df[df['fs']==fs]                   
                    axes[i,j].plot(df['iteration'], df[gf], c = colors[2], linewidth=2, label = 'Feedback Integration') #colors[j]    
                    ## draw cma line
                    if show_cma:
                        axes[i,j].plot(df['iteration'], df['CMA_'+gf], c = colors[0], linewidth=2, label = 'CMA of Feedback Integration')
                    ## draw baseline
                    df_p_null = df_group[df_group['participant_id'].isnull()]  
                    df = df_p_null[df_p_null['Feature']==sens_attr]
                    df = df[df['fs']==fs] 
                    axes[i,j].plot([l for l in range(xlim)], [df[gf].tolist()[0]]*xlim,c='black', label = 'Baseline - No Feedback')             
                    axes[i,j].set_xlabel("Iteration")
                    axes[i,j].set_ylabel(group_fair_codes[j])
                    axes[i,j].set_title(sens_attr)       
            ## ACCURACY
            df_p = df_acc[df_acc['participant_id']==p_id]
            ## draw time series
            df = df_p[df_p['fs']==fs]
            axes[len(sensitive_attrs),0].plot(df['iteration'], df['accuracy'], c = colors[2], linewidth=2)#colors[len(group_fair)+len(indiv_fair)]
            ## draw cma line
            if show_cma:
                axes[len(sensitive_attrs),0].plot(df['iteration'], df['CMA_accuracy'], c = colors[0], linewidth=2)
            ## draw baseline
            df = df_acc[df_acc['participant_id'].isnull()]
            df = df[df['fs']==fs]
            axes[len(sensitive_attrs),0].plot([l for l in range(xlim)], [df['accuracy'].tolist()[0]]*xlim, c = 'black')
            axes[0,0].legend(bbox_to_anchor=(0.0,2.0),loc='upper left') 
            axes[len(sensitive_attrs),0].set_xlabel("Iteration")
            axes[len(sensitive_attrs),0].set_ylabel('Accuracy %')
            fig.delaxes(axes[len(sensitive_attrs),1])
            
            fig.savefig(folder+'{}'.format(p_id)+image_type, dpi=300)
            plt.show()

def plot_silhouette_scores(silhouette_scores, parameters):
    matplotlib.rcParams.update({'font.size': 16})
    ## 
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
    fig.suptitle('Silhouette Score', fontweight='bold')
    ##
    axes.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    axes.set_xticks(range(len(silhouette_scores)))
    axes.set_xticklabels(list(parameters))
    # plt.title('Silhouette Score', fontweight='bold')
    axes.set_xlabel('Number of Clusters')
    plt.show()

def plot_clusters(array_of_vectors_pca, cluster_ids, colors):
    matplotlib.rcParams.update({'font.size': 16})
    n_clusters = len(np.unique(cluster_ids))
    max_cluster_id = max(cluster_ids)
    ## 
    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Clusters K-Means {}'.format(n_clusters), fontweight='bold')
    ##
    scat = axes.scatter(array_of_vectors_pca[:, 0], array_of_vectors_pca[:, 1], 
                c = cluster_ids, 
                cmap = matplotlib.colors.ListedColormap(colors[0:max_cluster_id+1]))
    cb = plt.colorbar(scat,ax=axes)
    loc = np.arange(0, max_cluster_id, max_cluster_id/float(max_cluster_id+1))
    cb.set_ticks(loc)
    cb.set_ticklabels(['cluster '+str(i) for i in range(max_cluster_id+1)])
    plt.show()

def perc_change_plots_per_cluster(perc_ch_df, cluster_df, title, file_name, folder, attrs, attrs_codes, group_fair, group_fair_codes):
    colors = plt.cm.tab10
    matplotlib.rcParams.update({'font.size': 30})
    ##
    min_v = {}
    max_v = {}
    for cl in cluster_df['cluster_id'].unique().tolist():
        if cl not in min_v:
            min_v[cl] = {}
            max_v[cl] = {}
        part_in_cl = cluster_df[cluster_df['cluster_id'] == cl]['participant_id'].tolist()
        for i,attr in enumerate(attrs):
            if attr not in min_v[cl]:
                min_v[cl][attr] = {}
                max_v[cl][attr] = {}
            for j,fm in enumerate(group_fair):
                if fm not in min_v[cl][attr]:
                    min_v[cl][attr][fm] = []
                    max_v[cl][attr][fm] = []
                df = perc_ch_df[perc_ch_df['participant_id'].isin(part_in_cl)][['participant_id',attr+'_'+fm]].sort_values(by=[attr+'_'+fm], ascending=False)
                min_v[cl][attr][fm].append(min(df[attr+'_'+fm].values.tolist()))
                max_v[cl][attr][fm].append(max(df[attr+'_'+fm].values.tolist()))
    for cl in cluster_df['cluster_id'].unique().tolist():        
        for i,attr in enumerate(attrs):            
            for j,fm in enumerate(group_fair):
                min_v[cl][attr][fm] = min(min_v[cl][attr][fm])
                max_v[cl][attr][fm] = max(max_v[cl][attr][fm])
    ##
    for cl in cluster_df['cluster_id'].unique().tolist():
        if len(group_fair) == 2 and 'DemographicParityRatio' in group_fair and 'AverageOddsDifference' in group_fair:
            fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 16), layout="constrained")
        else:
            fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 20), layout="constrained")
        fig.suptitle(title.format(str(cl)))
        part_in_cl = cluster_df[cluster_df['cluster_id'] == cl]['participant_id'].tolist()
        for i,attr in enumerate(attrs):
            for j,fm in enumerate(group_fair):
                df = perc_ch_df[perc_ch_df['participant_id'].isin(part_in_cl)][['participant_id',attr+'_'+fm]].sort_values(by=[attr+'_'+fm], ascending=False)            
                axes[j,i].bar(df['participant_id'], df[attr+'_'+fm], color = list(colors(cl+1)))
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(group_fair_codes[group_fair.index(fm)]+'\n Perc. Ch. %')
                axes[j,i].set_title(attrs_codes[i])
                axes[j,i].set_ylim(min_v[cl][attr][fm],max_v[cl][attr][fm])
                axes[j,i].xaxis.set_ticklabels([])
                if i ==0 and j==0:
                    handl, lab = axes[j,i].get_legend_handles_labels()
                    by_label = dict(zip(lab, handl))
                    by_label = dict(sorted(by_label.items()))
                    
        fig.savefig(folder+file_name.format(str(cl)), dpi=300)

def perc_change_plots(perc_ch_df, title, file_name, folder, attrs, attrs_codes, group_fair, group_fair_codes):
    colors = plt.cm.tab10
    matplotlib.rcParams.update({'font.size': 30})
    if len(group_fair) == 2 and 'DemographicParityRatio' in group_fair and 'AverageOddsDifference' in group_fair:
        fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 16), layout="constrained")
    else:
        fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 25), layout="constrained")
    fig.suptitle(title)
    for i,attr in enumerate(attrs):
        for j,fm in enumerate(group_fair):
            if fm == 'indiv.':
                if attr == 'CODE_GENDER':
                    fm = 'consistency_10'
                    code = ' (+)'
                elif attr =='NAME_FAMILY_STATUS':
                    fm = 'theil_index'
                    code = ' (-)'
                else:
                    fig.delaxes(axes[len(group_fair)-1,len(attrs)-1])
                    continue
                df = perc_ch_df[['participant_id',fm]].sort_values(by=[fm], ascending=False)                
                axes[j,i].bar(df['participant_id'], df[fm])
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(fm+code+'\n Perc. Ch. %')
                axes[j,i].xaxis.set_ticklabels([])
            else:
                df = perc_ch_df[['participant_id',attr+'_'+fm]].sort_values(by=[attr+'_'+fm], ascending=False)
                axes[j,i].bar(df['participant_id'], df[attr+'_'+fm])#,color=cls,label = labels
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(group_fair_codes[group_fair.index(fm)]+'\n Perc. Ch. %')
                axes[j,i].set_title(attrs_codes[i]) 
                axes[j,i].xaxis.set_ticklabels([])
    fig.savefig(folder+file_name, dpi=300)