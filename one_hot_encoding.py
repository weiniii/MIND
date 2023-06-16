import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

## import news
column_names = ['News_id', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title_Entities', 'Abstract_Entities']
news = pd.read_csv('./train//train_news.tsv', sep='\t', names=column_names)

## import behaviors
column_names = ['Impression_id', 'User', 'Time', 'Clicked_News', 'Impressions']
behaviors = pd.read_csv('./train/train_behaviors.tsv', sep='\t', names=column_names, parse_dates=['Time'])

## one hot encoding through subcategory
one_hot_subcategory = pd.get_dummies(pd.Series(news['Subcategory'])).astype(float).values
one_hot_total = one_hot_subcategory
one_hot = {news['News_id'][i]:one_hot_total[i] for i in range(len(news))}

## num variable
num_behaviors = len(behaviors)
one_hot_vector_size = len(pd.unique(news['Subcategory']))
num_impression = len(behaviors['Impressions'].values[0].split())

## user vector (clicked history) by one hot vector
user_click = np.zeros((num_behaviors, one_hot_vector_size))
for i in tqdm.tqdm(range(len(user_click))):
    for click in behaviors['Clicked_News'].values[i].split():
        user_click[i] += one_hot[click]

## impression vector by one hot vector   
impressions = np.empty((num_behaviors, num_impression, one_hot_vector_size))
for i in tqdm.tqdm(range(num_behaviors)):
    for j in range(num_impression):
        impression = behaviors['Impressions'].values[i].split()[j].split('-')[0]
        impressions[i, j] = one_hot[impression]

## probability by dot product between user vectors and impression vectors
prob = np.empty((num_behaviors, num_impression))
for i in tqdm.tqdm(range(num_behaviors)):
    prob[i] = impressions[i] @ user_click[i] 

## Truth label to verifying the performance
users_impressions_news_truth = []

for j in tqdm.tqdm(range(num_behaviors)):
    user_j = behaviors['Impressions'].values[j]
        
    user_j_truth_table = [float(user_j.split()[i].split('-')[1]) for i in range(len(user_j.split()))]
    
    users_impressions_news_truth.append(user_j_truth_table)  

## result
y = np.array(users_impressions_news_truth).reshape(-1)
x = prob.reshape(-1)
print(f'| ROC AUC Score: {roc_auc_score(y, x):5.3f} |') 