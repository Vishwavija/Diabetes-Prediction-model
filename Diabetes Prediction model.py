#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary liberary

# In[1]:


import numpy as np   
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from collections import OrderedDict 
import scipy.stats as sci
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score  
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report  
import warnings
warnings.filterwarnings('ignore')


# ## Importing  dataset
# 

# In[2]:


df = pd.read_csv('diabetes.csv')


# In[3]:


df.head(5)


# In[4]:


df.tail(5)


# ### exploratory data analysis (EDA)

# In[5]:


df.shape


# In[6]:


df.info()


# observation 
# 1.  Some of  are no null values 
# 2.  768 rows & 9 coloums 
# 3.  except BMI & Diabetestpedigreefunction all the coloums data type is int 
# 4.  Dependent variable is outcome and rest are independent variable.

# In[7]:


df.columns


# In[8]:


df.describe()


# Analysis from descriptive statistics
# 
# 1. There might be skewness in the data in the columns.
# 2. There might be chance of outliers if we compare Quartiles of some of the columns.(Pregnancies,Glucose,Insulin in the upper whisker region)
# 3. Since minimum and Q1 values are same for Skin_Thickness and Insulin  we do not have outliers in the Lower Whisker region for them.

# In[9]:


# Grouping the data by the 'Outcome' column and calculating the mean for each group
df.groupby('Outcome').mean()


# ## Bulding a custom summary function for EDA report

# In[10]:


def custom_summary(my_df):
    result = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'object':
            stats = OrderedDict({
                'Feature Name': col , 
                'Count': my_df[col].count() ,
                'Minimum': my_df[col].min() ,
                'Quartile1': my_df[col].quantile(.25) ,
                'Quartile2': my_df[col].quantile(.50) ,
                'Mean': my_df[col].mean() ,
                'Quartile 3': my_df[col].quantile(.75) ,
                'Maximum': my_df[col].max() ,
                'Variance': round(my_df[col].var()) ,
                'Standard Deviation': my_df[col].std() ,
                'Skewness': my_df[col].skew() , 
                'Kurtosis': my_df[col].kurt()
                })
            result.append(stats)
    result_df = pd.DataFrame(result)
    # skewness type
    skewness_label = []
    for i in result_df["Skewness"]:
        if i <= -1:
            skewness_label.append('Highly Negatively Skewed')
        elif -1 < i <= -0.5:
            skewness_label.append('Moderately Negatively Skewed')
        elif -0.5 < i < 0:
            skewness_label.append('Fairly Negatively Skewed')
        elif 0 <= i < 0.5:
            skewness_label.append('Fairly Positively Skewed')
        elif 0.5 <= i < 1:
            skewness_label.append('Moderately Positively Skewed')
        elif i >= 1:
            skewness_label.append('Highly Positively Skewed')
    result_df['Skewness Comment'] = skewness_label
    
    kurtosis_label=[]
    for i in result_df['Kurtosis']:
        if i >= 1:
            kurtosis_label.append('Leptokurtic Curve')
        elif i <= -1:
            kurtosis_label.append('Platykurtic Curve')
        else:
            kurtosis_label.append('Mesokurtic Curve')
    result_df['Kurtosis Comment'] = kurtosis_label
    Outliers_label = []
    for col in my_df.columns:
        if my_df[col].dtypes != 'object':
            Q1 = my_df[col].quantile(0.25)
            Q2 = my_df[col].quantile(0.5)
            Q3 = my_df[col].quantile(0.75)
            IQR = Q3 - Q1
            LW = Q1 - 1.5*IQR
            UW = Q3 + 1.5*IQR
            if len(my_df[(my_df[col] < LW) | (my_df[col] > UW)]) > 0:
                Outliers_label.append('Have Outliers')
            else:
                Outliers_label.append('No Outliers')
    result_df['Outlier Comment'] = Outliers_label

            
    return result_df


    

    


# In[11]:


custom_summary(df)


# In[12]:


def replace_outlier(my_df,col,method="Quartile",strategy="Median"):
    col_data = my_df[col] #method means how are you supposed to detect the outliers.
    
    if method == 'Quartile':
        #Using quartiles to calculate IQR
        q1 = col_data.quantile(0.25)
        q2 = col_data.quantile(0.5)
        q3 = col_data.quantile(0.75)

        IQR = q3 - q1
        LW = q1 - 1.5*IQR
        UW = q3 + 1.5*IQR
        
    elif method == 'Standard Deviation': #we are using empirical method here 
        mean = col_data.mean()
        std = col_data.std()
        LW = mean - 2*std
        UW = mean + 2*std
    else:
        print('Pass a correct method')
     
    ## printing all the outlier 
    outliers =  my_df.loc[(col_data < LW) | (col_data > UW)]
    outlier_density = round(len(outliers)/len(my_df) , 2) * 100 #What i am doing is, i am checking how many percentage of records are there which are outlier, so lets say if i have 15% are outliers so 15 % is my outliers density.
    if len(outliers) == 0:
        print(f'Feature {col} doesnot have any outliers')
        print('\n') #\n means next line
    else:
        print(f'Feature {col} Has outliers')
        print('\n')
        print(f'Total number of outliers in {col} are {len(outliers)}')
        print('\n')
        print(f'outlier percentage in {col} is {outlier_density}%')
        print('\n')
        display(my_df[(col_data < LW) | (col_data > UW)])
    
    ## Replacing outliers
    if strategy == 'Median':
        my_df.loc[(col_data < LW) | (col_data > UW) , col] = q2
    elif strategy == 'Mean':
         my_df.loc[(col_data < LW) | (col_data > UW) , col] = mean
    else:
        print('pass a correct strategy')
        
        
    return my_df

    
     


# In[13]:


def odt_plots(my_df, col):
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25,8))
    
    # Descriptive stats boxplot
    sns.boxplot(my_df[col] , ax = ax1)
    ax1.set_title(col + 'Boxplot')
    ax1.set_xlabel('values')
    ax1.set_ylabel('Boxplot')
    
    # Plotting histograms with outliers
    sns.distplot(my_df[col] , ax = ax2 , fit = sci.norm) # on histogram we are fitting normal distribution curve
    ax2.axvline(my_df[col].mean() , color = 'green')
    ax2.axvline(my_df[col].median() , color = 'brown')
    ax2.set_title(col + 'Histogram with outliers')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('values')
    
    #replacing outliers 
    df_out = replace_outlier(my_df , col)
    
    #Plotting Histogram without outliers
    sns.distplot(df_out[col] , ax = ax3, fit =sci.norm) #On histogram i am fitting normal Distribution plot
    ax3.axvline(df_out[col].mean() , color = 'green')
    ax3.axvline(df_out[col].median() , color = 'brown')
    ax3.set_title(col + 'Histogram without outliers')
    ax3.set_ylabel('Density')
    ax3.set_xlabel('values')
    plt.show()


    
    


# In[14]:


for col in df.columns:
    odt_plots(df, col)


# In[15]:


for col in df.columns:
    if col != 'Outcome':
        fig, ax1 = plt.subplots(figsize = (10,5))
        sns.regplot(x = df[col] , y = df['Outcome'] , ax = ax1).set_title(f'relationship between {col} and Outcome')
    


# Analysis from Regression plots:
# 
#    1.  Preganancies and outcome are strongly  positively correlated.
#    2. Glucose and outcome are stronly positively correlated.
#    3. Blood pressure and outcome are positively correlated.
#    4. skinThickness and outcome are slightly positively correlated.
#    5. Rest all othe independent variable are positively correlated with outcome.

# In[16]:


# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.title('Correlation Matrix Heatmap')
plt.show()


# Analysis
# 1.  Pregnancies has high correlation with Age
# 2.  Glucose has highest correlation with outcome
# 3.  BloodPressure has highest correlation with Age
# 4.  SkinThickness has highest correlation with Insulin
# 

# In[17]:


# Creating a pair plot to visualize pairwise relationships between variables, with 'Outcome' as the hue
sns.pairplot(df, hue='Outcome')

# Adding a title to the plot
plt.title('Pair Plot')

# Display the pair plot
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame containing the data

# Create a figure and axis objects
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the pie chart on the first axis (ax[0])
df['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Target')
ax[0].set_ylabel('')

# Plot the count plot on the second axis (ax[1])
sns.countplot('Outcome', data=df, ax=ax[1])
ax[1].set_title('Outcome')

# Show the plots
plt.show()


# ## Data Preprocessing
# 

# #####(2.1) Missing Observation Analysis¶
# 

# We saw on df.head() that some features contain 0,
# it doesn't make sense here and this indicates missing value Below we replace 0 value by NaN:

# In[19]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[20]:


df.head()


# In[21]:


df.isnull().sum()


# In[22]:


# The missing values will be filled with the median values of each variable.
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[23]:


# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[24]:


df.head()


# In[25]:


# Missing values were filled.
df.isnull().sum()


# ### Splitting our dataset
# 

# In[26]:


# Creating the feature variables by dropping the 'Outcome' column
X = df.drop(columns='Outcome', axis=1)

# Creating the target variable
Y = df['Outcome']


# In[27]:


# Displaying our feature variable
X


# In[28]:


# Displaying our target variable
Y


# ### Data Standardization
# 

# In[29]:


# Create an instance of the StandardScaler
scaler = StandardScaler()


# In[30]:


# Fitting the StandardScaler to the feature variables (X)
scaler.fit(X)


# In[31]:


# Transform the feature variables (X) using the fitted StandardScaler
standardized_df = scaler.transform(X)
standardized_df


# In[32]:


# Assigning the standardized feature variables to X
X = standardized_df

# Assigning the 'Outcome' column from the diabetes_dataset DataFrame to Y
Y = df['Outcome']


# In[33]:


# Displaying our feature variable after scaling
X


# In[34]:


# Displaying our target variable
Y


# ### Train Test Split
# 

# In[35]:


# Split the data into training and testing sets
# X_train: training feature variables
# X_test: testing feature variables
# Y_train: training target variable
# Y_test: testing target variable
# The data is split using a test size of 0.2 (20% of the data) and a random state of 2 for reproducibility

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[36]:


# Print the shapes of X, X_train, and X_test
print("Shape of X:", X.shape)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


# #### Training our Model
# 

# In[37]:


# Creating an instance of the Support Vector Classifier (SVC)
# The kernel parameter is set to 'rbf' indicating a radial basis function kernel
# The random_state parameter is set to 0 for reproducibility
SVM = SVC(kernel='rbf', random_state=0)


# In[38]:


# Fitting the SVM model to the training data
SVM.fit(X_train, Y_train)


# ### Model Evaluation

# ###### Accuracy Score of Training data

# In[39]:


# Calculate the accuracy score of the model on the training data
training_df_accuracy = SVM.score(X_train, Y_train)

# Print the accuracy score of the training data
print('Accuracy score of the training data:', training_df_accuracy)


# In[40]:


# Calculate the accuracy score of the model on the testing data
testing_df_accuracy = SVM.score(X_test, Y_test)

# Print the accuracy score of the testing data
print('Accuracy score of the testing data:', testing_df_accuracy)


# In[41]:


# Predicting the target variable for the testing data
y_predict = SVM.predict(X_test)

# Computing the confusion matrix
confusion_matrix(Y_test,y_predict)

# Creating a cross-tabulation table
pd.crosstab(Y_test, y_predict, rownames=['True'], colnames=['Predicted'], margins=True)


# In[42]:


#Computing the confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_predict)

# Creating a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')

# Setting the title of the heatmap
plt.title('Confusion Matrix', y=1.1)

# Setting the label for the y-axis
plt.ylabel('Actual label')

# Setting the label for the x-axis
plt.xlabel('Predicted label')

# Displaying the heatmap
plt.show()


# #### Classification Report
# 

# In[43]:


print(classification_report(Y_test,y_predict))


# ###  Predictive System
# 

# In[44]:


# Defining the input data
input_df = (4, 113, 74, 2, 173, 25.8, 0.66, 23)

# Convert the input data to a numpy array
input_df_as_numpy_array = np.asarray(input_df)

# Reshape the array as we are predicting for one instance
input_df_reshaped = input_df_as_numpy_array.reshape(1, -1)

# Standardize the input data
std_df = scaler.transform(input_df_reshaped)
print("Standardized input data:", std_df)

# Make the prediction using the SVM model
prediction = SVM.predict(std_df)
print("Prediction:", prediction)

# Print the result based on the prediction
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:




