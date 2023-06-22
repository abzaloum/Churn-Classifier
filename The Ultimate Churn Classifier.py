#!/usr/bin/env python
# coding: utf-8

# In[1]:


#The purpose of these classifiers is to find behaviors correlated with churn vs those correlated with retention.
#There are two classifiers below, Random Forest and XGBoost


# In[2]:


import pandas as pd


# In[7]:


#get data into pandas dataframe
df = pd.read_csv('/Users/alexzaloum/Downloads/data_for_churn_classifier.csv')


# In[8]:


df_list = df.values.tolist()


# In[9]:


df = df.iloc[:, ::-1]
print(df.columns)
print(df.shape)

#Drop the columns that are not suitable or you do not wish to include as features in the classifier. Keep retention_time, which will be dropped below.
df = df.drop(columns = ['user_id','registration_date','user_earliest_activity_date','user_latest_activity_date', 'Sum of New Masked Email Created (Email platform)']) #,'New Masked Email Created (Email platform)']) #, 'Indicator New Masked Email Created (Email platform)'])


# In[5]:


#Slicing the data into two sets: churned and retained users
#Select the number of days the users were seen or active (i.e. retained) after registration, that will signify whether the users "churned" or were "retained"

days_after = 1     #7, 30, etc

retained = df[df['retention_time']> days_after]
churned = df[df['retention_time']<= days_after]
retained = retained.drop(columns = ['retention_time'])
churned = churned.drop(columns = ['retention_time'])

print(retained.shape)
print(churned.shape)

#retained = df[df["time retained (active)"] > 1]
#retained = retained.drop(columns = ['mobile extension install',
#       'web extension install','retained time (seen)', 'LAST_SEEN',
#       'time retained (active)', 'LAST_ACTIVE',
#       'ORIGINAL_CLIENT_APPLICATION_ID', 'REGISTRATION_DATE', 'ID'])
##retained = retained.drop(columns = ['id','registration_date', 'original_client_application_id', 'last_active', 'time retained (active)', 'last_seen', 'retained time (seen)', 'blur_extension_install_date','blur_mobile_install_date'])
#print(retained.shape)
#churned = df[df["time retained (active)"] <= 1]
#churned = churned.drop(columns = ['mobile extension install',
#       'web extension install','retained time (seen)', 'LAST_SEEN',
#       'time retained (active)', 'LAST_ACTIVE',
#       'ORIGINAL_CLIENT_APPLICATION_ID', 'REGISTRATION_DATE', 'ID'])
##churned = churned.drop(columns = ['id','registration_date', 'original_client_application_id', 'last_active', 'time retained (active)', 'last_seen', 'retained time (seen)', 'blur_extension_install_date','blur_mobile_install_date'])
#print(churned.shape)


# In[10]:


#Concatenate retained and churned sets into input features X and and output labels y
X = pd.concat([retained, churned])
y = [1] * retained.shape[0] + [0] * churned.shape[0]


# In[7]:


#Preparing the data for the classifier

#from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import sklearn.datasets
from sklearn.preprocessing import OneHotEncoder

randomstate = 10

#X, y = make_classification(
#    n_samples=40000,
#    n_features=29,
#    n_informative=29,
#    n_redundant=0,
#    n_repeated=0,
#    n_classes=2,
#    random_state=randomstate,
#    shuffle=False,
#)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=randomstate)

print(X_train.shape[0])

#We need to encode nominal (non-numeric) columns in order to feed them into the classifier
columns_to_encode = ['browser', 'country', 'product_group', 'device_type'] 

# Create the encoder.
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X_train[columns_to_encode])    # Assume for simplicity all features are categorical.

# Apply the encoder.
X_train_encoded = encoder.transform(X_train[columns_to_encode]).toarray()
X_test_encoded = encoder.transform(X_test[columns_to_encode]).toarray()


#from sklearn.preprocessing import OneHotEncoder
#
## Create the encoder.
#encoder = OneHotEncoder(sparse = False, handle_unknown="ignore")
#encoder.fit(X_train)    # Assume for simplicity all features are categorical.
#
## Apply the encoder.
#X_train = encoder.transform(X_train)
#X_test = encoder.transform(X_test)

#transformed = jobs_encoder.transform(data['Profession'].to_numpy().reshape(-1, 1))
##Create a Pandas DataFrame of the hot encoded column
#ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names())
##concat with original data
#data = pd.concat([data, ohe_df], axis=1).drop(['Profession'], axis=1)
#

#print(len(X_train.iloc[0]))

encoder.categories_
encoded_columns = np.concatenate(encoder.categories_)
#print(encoded_columns)

#print(type(np.array(X_train.drop(columns = columns_to_encode).columns)))
#print(type(encoded_columns))
#
column_names = np.concatenate((np.array(X_train.drop(columns = columns_to_encode).columns), encoded_columns))
#print(column_names)

#convert np-array back to dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoded_columns)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoded_columns)

X_train_not_encoded = X_train.drop(columns = columns_to_encode)
X_test_not_encoded = X_test.drop(columns = columns_to_encode)


X_train = pd.concat([X_train_not_encoded.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = "columns")
X_test = pd.concat([X_test_not_encoded.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = "columns")

#to just run with nominal features
#X_train = X_train_encoded
#X_test = X_test_encoded

#to just run with numerical features
#X_train = X_train_not_encoded
#X_test = X_test_not_encoded

#print(X_train)

#standardization and normalization not technically required for Random Forest but just to check
#X_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
#X_test = pd.DataFrame(StandardScaler().fit_transform(X_test), columns=X_test.columns)
#
#X_train = pd.DataFrame(Normalizer().fit_transform(X_train), columns=X_train.columns)
#X_test = pd.DataFrame(Normalizer().fit_transform(X_test), columns=X_test.columns)
#
#(Why is normalize different from Normalizer?)
#X_train = pd.DataFrame(sklearn.preprocessing.normalize(X_train, norm='l2'))
#X_test = pd.DataFrame(sklearn.preprocessing.normalize(X_test, norm='l2'))

print(X_train.columns)


# In[8]:


#Run Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
feature_names = [str(X_train.columns[i]) for i in range(len(X_train.columns))]
forest = RandomForestClassifier(n_estimators = 1000, random_state= randomstate)
forest.fit(X_train, y_train)


# In[9]:


#test the time it takes classifier to run
import time
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[10]:


# get the "importance" i.e. contribution of each feature to the classifier

import pandas as pd
import matplotlib.pyplot as plt
#zipped = zip(feature_names,importances.tolist())
#feature_importances_sorted = sorted(zipped, key=lambda x: x[1], reverse= True)
#print(feature_importances_sorted)
#top_important_features = feature_importances_sorted[0:10]
#top_importances = [top_important_features[i][1] for i in range(0,10)]
#top_feature_names = [top_important_features[i][0] for i in range(0,10)]

forest_importances = pd.Series(importances, index=feature_names)


print(feature_names)

#fig, ax = plt.subplots(figsize=(15, 15))
#forest_importances.plot.bar(yerr=std, ax=ax)
#ax.set_title("Feature importances using MDI")
#ax.set_ylabel("Mean decrease in impurity")
#fig.tight_layout()

import shap

explainer = shap.TreeExplainer(forest)

shap_values = np.array(explainer.shap_values(X_train))
print(shap_values.shape)
print(shap_values[1].shape)#sanity check. see: https://stackoverflow.com/questions/65549588/shap-treeexplainer-for-randomforest-multiclass-what-is-shap-valuesi
shap_values_ = shap_values.transpose((1,0,2))

np.allclose(
    forest.predict_proba(X_train),
    shap_values_.sum(2) + explainer.expected_value
)

shap.summary_plot(shap_values[1],X_train, feature_names = feature_names)
shap_values = explainer.shap_values(X_test)
#
#
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names = feature_names)
#
#rf_resultX = pd.DataFrame([shap_values, feature_names])
#
#rf_resultX = pd.DataFrame(shap_values, columns = feature_names)
#
#vals = np.abs(rf_resultX.values).mean(0)
#
#shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
#                                  columns=['col_name','feature_importance_vals'])
#shap_importance.sort_values(by=['feature_importance_vals'],
#                               ascending=False, inplace=True)
#shap_importance.head()
# ## from sklearn.inspection import permutation_importance
# 
# start_time = time.time()
# result = permutation_importance(
#     forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
# 
# forest_importances = pd.Series(result.importances_mean, index=feature_names)

# In[11]:


#plot and print the importances in descending order, most important feature first
fig, ax = plt.subplots(figsize=(15, 15))
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
#plt.show()
print(forest_importances.sort_values(ascending=False)[0:30])


# In[12]:


#Here we can view individual features and see its breakdown between churned and retained.
#Keep in mind because we encoded country, each country is a separate feature, so we can view e.g. Mexico by itself

top_features_list = list(forest_importances.sort_values(ascending=False).index)
X_combined = pd.concat([X_train.reset_index(drop=True), X_test.reset_index(drop=True)], axis = "rows")
y_df = pd.DataFrame(y)
X_y_combined = pd.concat([X_combined.reset_index(drop=True), y_df.reset_index(drop=True)], axis = "columns")

select_feature = "Mexico" #top_features_list[-2]

X_y_combined_selected_feature = X_y_combined[X_y_combined[select_feature]>0]

#feature_indices = X_train[X_train[top_features_list[0]]
x_axis = [0,1]
y_axis = [X_y_combined_selected_feature[X_y_combined_selected_feature[0] == 0].shape[0], X_y_combined_selected_feature[X_y_combined_selected_feature[0] == 1].shape[0]]

plt.bar(x_axis, y_axis)
plt.xlabel("churn / retain")
plt.ylabel("user count")
plt.title(select_feature)
print(select_feature)
print(str(round(y_axis[1]/(y_axis[0]+y_axis[1])*100,2)) + str('% retained'))


# In[13]:


#Retention rate but user volume can give us a sense of where to focus our efforts. 
#Focus on places where rates are lower and volume is higher

retention_rates = []
feature_volumes = []

print(top_features_list[53])
#X_y_combined = X_y_combined[~X_y_combined.columns.duplicated()]
#print(X_y_combined[X_y_combined.index.duplicated()])

country_list = ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Anonymous Proxy', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bolivia', 'Bosnia and Herzegovina', 'Brazil', 'British Virgin Islands', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Chile', 'China', 'Colombia', 'Congo', 'Costa Rica', 'Croatia', 'Curaçao', 'Cyprus', 'Czech Republic', 'Czechia', 'Denmark', 'Djibouti', 'Dominican Republic', 'Ecuador', 'Egypt', 'Estonia', 'Ethiopia', 'Europe', 'Fiji', 'Finland', 'France', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Hashemite Kingdom of Jordan', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Libya', 'Libyan Arab Jamahiriya', 'Lithuania', 'Luxembourg', 'Macao', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Malta', 'Mauritius', 'Mexico', 'Missing_country', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Nigeria', 'No Country', 'Norway', 'Oman', 'Pakistan', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Republic of Korea', 'Republic of Lithuania', 'Republic of Moldova', 'Reunion', 'Romania', 'Russia', 'Russian Federation', 'Rwanda', 'Saint Lucia', 'Satellite Provider', 'Saudi Arabia', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Taiwan, Province of China', 'Tajikistan', 'Tanzania', 'Thailand', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'U.S. Virgin Islands', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Unknown', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']


N = []

for i in range(len(top_features_list)):
    if top_features_list[i] in country_list:
        N.append(i)

top_country_list = [top_features_list[i] for i in N]
        
number_of_countries_to_visualize = 10
    
for i in N[0:number_of_countries_to_visualize]:
    select_feature = top_features_list[i]
    X_y_combined_selected_feature = X_y_combined[X_y_combined[select_feature]>0]
    y_axis = [X_y_combined_selected_feature[X_y_combined_selected_feature[0] == 0].shape[0], X_y_combined_selected_feature[X_y_combined_selected_feature[0] == 1].shape[0]]
    if y_axis[0]+y_axis[1] > 0:
        retention_rates.append(round(y_axis[1]/(y_axis[0]+y_axis[1])*100,2))
    else:
        retention_rates.append(0)
    feature_volumes.append(X_y_combined[X_y_combined[select_feature]>0].shape[0])

print(type(retention_rates))



fig, ax = plt.subplots()
ax.scatter(retention_rates, feature_volumes)
ax.set_xlabel("retention rate")
ax.set_ylabel("user volume")

for i, txt in enumerate(top_country_list[0:number_of_countries_to_visualize]):
    ax.annotate(txt, (retention_rates[i], feature_volumes[i]))
    


                

#create scatter plot of assists vs. points
#ax = df.plot(kind='scatter', x='assists', y='points')
#
##label each point in scatter plot
#for idx, row in df.iterrows():
    #ax.annotate(row['team'], (row['assists'], row['points']))

from sklearn.inspection import permutation_importance
r = permutation_importance(forest, X_test, y_test,
                           n_repeats=10,
                           random_state=randomstate)
perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in pd.DataFrame(X_train).columns])
perm['AVG_Importance'] = r.importances_meanfig, ax = plt.subplots(figsize=(15, 15))
perm.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
print(perm)
# In[13]:


#This gives the overall accuracy of the classifier in distinguishing between churn and retained users on the test set given the training set
forest.score(X_test,y_test)


# In[14]:


#A confusion matrix gives a clear breakdown of the FP, TP, FN, and TNs

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_test = forest.predict(X_test)
confusion_matrix(y_test, y_pred_test)


# In[16]:


import seaborn as sns

# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Retained', 'Churned']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[17]:



from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 1000))
sel.fit(X_train, y_train)
sel.get_support()
print(list(sel.get_support()))
selected_feat= pd.DataFrame(X_train).columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)


# In[15]:


#GRADIENT BOOST CLASSIFIER
#Now we do the same thing for a Boost classifier as for Random Forest
#If not all steps from above are exactly replicated below, they can be with relative ease (most of them already are)


# In[19]:


from sklearn.ensemble import GradientBoostingClassifier
feature_names = [str(X_train.columns[i]) for i in range(len(X_train.columns))]
print(feature_names)
Boost = GradientBoostingClassifier(n_estimators=200, random_state=randomstate).fit(X_train, y_train)
print(Boost.score(X_test, y_test))
print(Boost.feature_importances_)
#print(clf.get_params(deep=True))


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
#zipped = zip(feature_names,importances.tolist())
#feature_importances_sorted = sorted(zipped, key=lambda x: x[1], reverse= True)
#print(feature_importances_sorted)
#top_important_features = feature_importances_sorted[0:10]
#top_importances = [top_important_features[i][1] for i in range(0,10)]
#top_feature_names = [top_important_features[i][0] for i in range(0,10)]

Boost_feature_importances = pd.Series(Boost.feature_importances_, index=feature_names)

fig, ax = plt.subplots(figsize=(15, 15))
Boost_feature_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
#fig.tight_layout()
print(Boost_feature_importances.sort_values(ascending=False)[0:30])


# In[21]:


from sklearn.inspection import permutation_importance
r = permutation_importance(Boost, X_test, y_test,
                           n_repeats=10,
                           random_state=randomstate)
perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in pd.DataFrame(X_train).columns])
perm['AVG_Importance'] = r.importances_mean


# In[22]:


fig, ax = plt.subplots(figsize=(15, 15))
perm.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()


# In[23]:


print('XGBoost feature importances')
print(Boost_feature_importances.sort_values(ascending=False))

print('\n permutation importances')
print(perm['AVG_Importance'].sort_values(ascending=False))

print('\n\n Overall model accuracy:' + str(round(Boost.score(X_test,y_test),3)))


# In[24]:


y_pred_test = Boost.predict(X_test)
confusion_matrix(y_test, y_pred_test)
# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Retained', 'Churned']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# In[25]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
sel.get_support()
print(list(sel.get_support()))
selected_feat= pd.DataFrame(X_train).columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)


# In[ ]:




