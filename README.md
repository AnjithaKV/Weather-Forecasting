# Weather-Forecasting

# Overview
This project aims to predict rainfall patterns using machine learning. The dataset was obtained from Kaggle and includes various features related to weather conditions. The project involves data preprocessing, feature selection, model training, and evaluation to predict rainfall accurately.
# Features
1) Utilizes historical weather data from Kaggle.
2) Implements machine learning algorithms including KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, and ensemble techniques for rainfall prediction.
2) Employs various visualization techniques such as pie chart, countplot, heatmap, box plot, and pairplot for data exploration and analysis.
3) Handles missing values using appropriate techniques.
4) Encodes categorical features using Label Encoder.
5) Performs feature selection using Mutual Information Gain and SelectKBest.
6) Removes outliers using box plot and IQR method.
7) Conducts oversampling using SMOTE to handle class imbalance.
8) Splits data into training and testing sets using StratifiedShuffleSplit.
9) Normalizes features using StandardScaler.
# Required Python libraries:
1) NumPy
2) Pandas
3) Matplotlib
4) Seaborn
5) Scikit-learn
6) Streamlit

# Visualization
1) Pie Chart: Visualizes the distribution of classes.
2) Countplot: Displays the distribution of categorical variables.
3) Heatmap: Shows the correlation between different features.
4) Box Plot: Identifies outliers in the dataset.
5) Pairplot: Visualizes the relationship between different features.

Model Performance
The performance metrics (Accuracy, Precision, Recall) of various models are as follows:

# Model:	 
The Accuracy,	Precision and	Recall of the models are respectively given below
1) KNN :                   85	      80	      92
2) Naive Bayes:	              73	      79      	64
3) Decision Tree:	            88	      90	      86
4) Random Forest:	            78       	80       	73
5) XGBoost	:                  90      	93      	87
6) Tuning XGBoost:	          90      	93      	87
7) Bagging DecisionTree:	    90	      90	      90
8) Bagging XGBoost	    :      90	      93	      87

Overall, this project provides valuable insights into the application of machine learning in addressing real-world challenges such as rainfall prediction, with potential applications in various fields such as agriculture, urban planning, and disaster management. 
