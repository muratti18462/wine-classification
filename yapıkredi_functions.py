import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 
from tabulate import tabulate
import scipy.stats as stats

"""
excepts 4 parameters: 
    first one is dataframe which will be worked on 
    second and third are the columns
    last parameter is for running little's mcar test
"""
def createTable(data,col1,col2, little = False):
    df = data.copy()
    df['NUMBER'] = 1
    if little:
        df['NULL_FLAG'] = np.where(df[col1].isnull() == True,'NULL', 'NOT_NULL')
    else: 
        df['NULL_FLAG'] = df[col1]
            
    table = pd.pivot_table(df,values = 'NUMBER',index=['NULL_FLAG'],columns=[col2], aggfunc=np.sum,fill_value=0)
    table = table.rename_axis(None)
    del df
    return table

"""
accepts 5 parameters:
    first one is dataframe which will be worked on 
    second and third are the columns
    fourth is if user wants to run little's  mcar test, default is False
    last is do_plot for barplottting categoric vs categoric columns distribution
"""
def catCatTest(data, col1 ,col2 , little = False, do_plot = False,do_eda = False):
  
    table = createTable(data,col1,col2, little)
    heads = data[col2].unique()
    heads.sort()
    print(tabulate(table,headers=heads))
    print('Statistics for ', col2)
    
    inp = 'DEPENDENT'
    X2, p, dof, expected = stats.chi2_contingency(table)
    n = np.sum(table).sum()
    minDim = min(table.shape)-1

#calculate Cramer's V 
    V = np.sqrt((X2/n) / minDim)
    
    if p > 0.05:
        inp = 'INDEPENDENT'
        mcar = 'RANDOM'
    else:
        if  V > 0.5:
            mcar = 'HIGH ASSOCIATION'
        elif  V > 0.3:
            mcar = 'MODERATE ASSOCIATION'
        else  :
            mcar = 'LOW ASSOCIATION'
      
    print('-'*80)
    print('p value: ', p ,'and Crammers V:', V)
    print('Dependency ? :', inp, 'and Association ?:', mcar)
    print('='*80)
    del table
    
    
def pie_plot(df,col):
    v_counts = df[col].value_counts()
    colors=cm.Set1(np.arange(len(v_counts))/float(len(v_counts)))
    fig = plt.figure(figsize=(8,8))
    plt.pie(v_counts, labels=v_counts.index, colors = colors, autopct='%1.1f%%', shadow=True);    
    
#%% Continues feature dist, prob, and box plot    
def contPlot(data,col):
  fig, axes = plt.subplots(1,3,figsize=(18,6))
  plt.subplot(1, 3, 1)
  sns.distplot(data[col],bins =50)
  plt.subplot(1, 3, 2)
  stats.probplot(data[col], dist="norm", plot=plt)
  plt.title(col)
  plt.subplot(1,3,3)
  sns.boxplot(data[col])
  plt.show()

## Check whether categoric features has effect on income statistically
def CatColumnRelation(data,col):
    data['NUMBER'] = 1
    table = pd.pivot_table(data,values = 'NUMBER',index=[col],columns=['income'], aggfunc=np.sum,fill_value=0).rename_axis(None)

    X2, p, dof, expected = stats.chi2_contingency(table)
    n = np.sum(table).sum()
    minDim = min(table.shape)-1

    #calculate Cramer's V 
    V = np.sqrt((X2/n) / minDim)
    data.drop(columns = 'NUMBER',inplace = True)
    return X2,p,V

def CatCounts(data,col):
    data['total_count'] = 1
    total = data['total_count'].sum()

    group_total = data.groupby([col])['total_count'].count().reset_index()
    group_total.rename(columns = {'total_count':'Group_Count'},inplace = True)

    temp = pd.DataFrame(data.groupby([col,'income'])['total_count'].count()).reset_index()
    temp['% of Total'] = ((temp.total_count / total)*100).round(2)
    temp.rename(columns = {'total_count':'Count'},inplace = True)

    temp = pd.merge(temp,group_total, on =col)
    temp['% of Group'] = (temp['Count']/ temp['Group_Count']*100).round(2)
    temp.drop(columns = 'Group_Count',inplace = True)
    temp = temp[[col, 'income','Count','% of Group','% of Total']]
    return temp

## For Categoric Columns
def catColumnInspector(data,col):
    data['total_count'] = 1
    X2,p,V = CatColumnRelation(data,col)
    
    print('\033[1m\t\t' + col + '\033[0m')
    print('-'*60)
    print('Statistical Test Results')
    print('p-value: ', p.round(5), ' Chi-Square: ', X2.round(2), " Crammer's V: ",V.round(5))
    print('-'*60)
    print(tabulate(CatCounts(data,col),headers=[col,'income','Count','% of Group','% of Total']))
    print('-'*60)
    print('')
    #mean income rate for each group
    mean_income = data.groupby([col])['income'].mean().reset_index()
    mean_income[col] = mean_income[col].astype('object')
    
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()
    
    sns.barplot(data=pd.DataFrame(data.groupby([col,'income'])['total_count'].count()).reset_index(),
                x=col, y='total_count', hue='income', ax=ax1)
    mean_income.plot(x=col, y='income', ax=ax2, color='r', linestyle='-', label='avg-income-rate')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.tick_params(axis='x', rotation=45)
    
    plt.show()
    data.drop(columns='total_count',inplace = True)
    
## For Numeric Columns
def numColumnInspector(data,col):
    data['total_count'] = 1
    ph_cor , p = stats.pointbiserialr(data['class'], data[col])
    print('\033[1m  \t\t\t' + col + '\033[0m')
    print('-'*75)
    print('Point Biseral Correlation Results ')
    print('Correlation: ', ph_cor.round(3), ' and  p-value: ', p)    
    print('-'*75)
    print(tabulate(data.groupby('class')[col].describe(),headers=['class','count','mean','std','min','25%','50%','75%','max']))
    print('-'*75)
    print('')
    plt.figure(figsize=(20,10))
    sns.boxplot(x=data["class"],y=data[col], showmeans=True)
    plt.show()       
    data.drop(columns=['total_count'],inplace = True)
   