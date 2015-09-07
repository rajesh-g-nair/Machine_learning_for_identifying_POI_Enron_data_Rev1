#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Function to Plot graph to visualize the data points
def plotGraph(data1,Label1, Label2, indx1=None, indx2=None, indx3=None):
    plt = matplotlib.pyplot
    for point in data1:    
        if indx3 == None:
            plt.scatter( point[indx1], point[indx2])         
        else:
            if point[indx3] == 1:
                plt.scatter( point[indx1], point[indx2], color = "r", label = "poi")        
            else:
                plt.scatter( point[indx1], point[indx2], color = "b", label = "non-poi")        
    plt.xlabel(Label1)
    plt.ylabel(Label2)
    plt.show()        

### Validation and Selection of the Best Suited Classifier
### Stratified Suffle Split Cross Validation has been used for getting the training and test set
### GridSearchCV used for best Estimator selection
def validate_classifier(svr,parameters,score, dataset, feature_list, folds = 1000, clf_name = None):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    if clf_name == 'NaiveBayes':    
        clf = svr
    else:
        from sklearn.grid_search import GridSearchCV    
        ### Using GridSeqrchCV to arrive at the best estimator for the classifier
        grid = GridSearchCV(svr,parameters,cv=cv,scoring=score)    
        grid.fit(features,labels)  
        print("The best parameters are %s with a score of %0.2f"
                    % (grid.best_params_, grid.best_score_))
        print("Grid scores on the given data set:")    
        for params, mean_score, scores in grid.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                 % (mean_score, scores.std() / 2, params))
        ### Assign the best estimator arrived using the Grid Scores as the Classifier         
        clf = grid.best_estimator_         
    feature_imp = []   
    count = 1
    labels_cfm = []
    predict_cfm = []
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        ### fit the classifier using training set, and test on test set        
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        if count == 1:
            labels_cfm = labels_test
            predict_cfm = predictions
        else:
            labels_cfm = np.concatenate((labels_cfm,labels_test))
            predict_cfm= np.concatenate((predict_cfm,predictions))
        if clf_name == "DecisionTreeClassifier":
            ### For each iteration store the feature importances
            for i in range(0,len(clf.feature_importances_)):
                if count == 1:
                    feature_imp.append(clf.feature_importances_[i])
                else:
                    feature_imp[i] = feature_imp[i] + clf.feature_importances_[i]
        count = count + 1
    if clf_name == "DecisionTreeClassifier":
        ### Print the feature importances by averaging for all the iterations
        for i in range(0,len(feature_list)-1):        
            print "importance of the feature ", feature_list[i+1], ":" , feature_imp[i] / count
    try:
        print clf        
        total_predictions = len(predict_cfm)
        print "Total Predictions: ", total_predictions
        from sklearn.metrics import confusion_matrix, precision_score, recall_score,  fbeta_score, accuracy_score
        cfm = confusion_matrix(labels_cfm,predict_cfm)
        print "Confusion Matrix:\n", cfm
        print "Accuracy:", round(accuracy_score(labels_cfm,predict_cfm),5)        
        print "Precision:", round(precision_score(labels_cfm,predict_cfm),5)
        print "Recall:", round(recall_score(labels_cfm,predict_cfm),5)
        print "F1 Score:", round(fbeta_score(labels_cfm, predict_cfm,1),5)
        print "F2 Score:", round(fbeta_score(labels_cfm, predict_cfm,2),5)        
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
    return clf

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
### Step 1: Identify Outliers
data = featureFormat(data_dict, features_list)
plotGraph(data,"Total_Stock_Value", "total_payments", 8,3,0)
### Step 2: Remove Outliers
### Remove the Outlier "TOTAL"
data_dict.pop("TOTAL",0) 
### Remove the data point "THE TRAVEL AGENCY IN THE PARK" as it has only the "Other" 
### and "Total Payments" populated and rest of the fields are not available
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0) 

### Task 3: Create new feature(s)
for key in data_dict:
    ### New Feature for Shared receipt as a fraction of total messages received     
    fraction_shared_receipt_with_poi = float(data_dict[key]['shared_receipt_with_poi']) / float(data_dict[key]['to_messages'])    
    if np.isnan(fraction_shared_receipt_with_poi):     
        data_dict[key]['fraction_shared_receipt_with_poi'] =  0
    else:
        data_dict[key]['fraction_shared_receipt_with_poi'] =  round(fraction_shared_receipt_with_poi,2)
    ### New Feature for fraction of messages received from poi     
    fraction_from_poi = float(data_dict[key]['from_poi_to_this_person']) / float(data_dict[key]['to_messages']) 
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_from_poi'] =  0
    else:
        data_dict[key]['fraction_from_poi'] =  round(fraction_from_poi,2)
     ### New Feature for fraction of messages sent to poi
    fraction_to_poi = float(data_dict[key]['from_this_person_to_poi']) / float(data_dict[key]['from_messages'])
    if np.isnan(fraction_from_poi):     
        data_dict[key]['fraction_to_poi'] =  0
    else:
        data_dict[key]['fraction_to_poi'] =  round(fraction_to_poi,2)
    
### Select K-Best features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.feature_selection import SelectKBest, f_classif
selector=SelectKBest(f_classif,k=22)
selector.fit(features,labels)
### print the features and their indices
print "The features to be used for the Classification is as below: "
for i in list(selector.get_support(indices=True)):
    print features_list[i + 1], ' - Feature Index: ', i + 1
print " "    

### Plotting the 6 Best features identified  
plotGraph(data,"Exercised Stock Options","fraction_to_poi", 10, 22, 0)
plotGraph(data,"bonus","shared_receipt_with_poi", 5, 19, 0)
plotGraph(data,"expenses","other", 9, 11, 0)

### Revised feature list containing the 5 best features having the discriminating power
features_list = ['poi','exercised_stock_options','fraction_to_poi','bonus','shared_receipt_with_poi','expenses']

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Uncomment to set the algorithm of the classifier to Gaussian Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#svr = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
#parameters=None
#svr_name = "NaiveBayes"

### Uncomment to set the algorithm of the classifier to SVC
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.svm import SVC
#from sklearn.pipeline import Pipeline
#parameters = dict(SVC__kernel=['linear','rbf'],SVC__C=[1,10,100,1000],SVC__gamma=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
#estimators = [('scale', MinMaxScaler()), ('SVC', SVC())]
#svr = Pipeline(estimators)
#svr_name = "SVC"

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
parameters = [{'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]
svr = DecisionTreeClassifier()    
svr_name = "DecisionTreeClassifier"


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

score = 'f1'
clf = validate_classifier(svr,parameters,score, my_dataset, features_list, folds = 1000,clf_name = svr_name)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)