# -*- coding: utf-8 -*-
"""
Created on Sun Feb 04 15:11:36 2018

@author: david
"""


import matplotlib as mpl;
import matplotlib.pyplot as plt;

import numpy as np;
import gzip;
import StringIO

#%%
        # ----------------------------- Import Labels and Features -------------------------------------


def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')];
    columns = headline.split(',');

    # The first column should be timestamp:
    assert columns[0] == 'timestamp';
    # The last column should be label_source:
    assert columns[-1] == 'label_source';
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:','');
        pass;
    
    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO.StringIO(csv_str),delimiter=',',skiprows=1);
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps);

'''
Read the data (precomputed sensor-features and labels) for a useuuid = '1155FF54-63D3-4AB2-9863-8385D0BD0A13';
(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid);r.
This function assumes the user's data file is present.
'''
def read_user_data(uuid):
    user_data_file = '%s.features_labels.csv.gz' % uuid;

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read();

    feature_names, label_names = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);

    return (X,Y,M,timestamps,feature_names,label_names);

uuid =[]
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/00EABED2-271D-49D8-B599-1D4A09240601')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/0A986513-7828-4D53-AA1F-E02D6DF9561B')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/0BFC35E2-4817-4865-BFA7-764742302A2D')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/0E6184E1-90C0-48EE-B25A-F1ECB7B9714E')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/2C32C23E-E30C-498A-8DD2-0EFB9150A02E')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/4E98F91F-4654-42EF-B908-A3389443F2E7')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/4FC32141-E888-4BFF-8804-12559A491D8C')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/5EF64122-B513-46AE-BCF1-E62AAC285D2C')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/7CE37510-56D0-4120-A1CF-0E23351428D2')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/9DC38D04-E82E-4F29-AB52-B476535226F2')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/11B5EC4D-4133-4289-B475-4E737182A406')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/24E40C4C-A349-4F9F-93AB-01D00FB994AF')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/27E04243-B138-4F40-A164-F40B60165CF3')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/33A85C34-CFE4-4732-9E73-0A7AC861B27A')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/40E170A7-607B-4578-AF04-F021C3B0384A')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/74B86067-5D4B-43CF-82CF-341B76BEA0F4')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/78A91A4E-4A51-4065-BDA7-94755F0BB3BB')
uuid.append('F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/83CF687B-7CEC-434B-9FE8-00C3D5799BE6')


(X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid[0])
for i in range (1,8):   
    (X_temp,Y_temp,M_temp,timestamps_temp,feature_names_temp,label_names_temp) \
      = read_user_data(uuid[i])
      
    X = np.vstack((X, X_temp))
    Y = np.vstack((Y, Y_temp))
    M = np.vstack((M, M_temp))
    timestamps = np.hstack((timestamps, timestamps_temp))


#%%
        # ----------------------------- Set Label Names -------------------------------------


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label;

n_examples_per_label = np.sum(Y,axis=0);
labels_and_counts = zip(label_names,n_examples_per_label);
sorted_labels_and_counts = sorted(labels_and_counts,reverse=True,key=lambda pair:pair[1]);
print "How many examples does this user have for each contex-label:";
print "-"*20;
for (label,count) in sorted_labels_and_counts:
    print "%s - %d minutes" % (get_label_pretty_name(label),count);
    pass;
    
print "User %s has %d examples (~%d minutes of behavior)" % (uuid,len(timestamps),len(timestamps));
timestamps.shape
print "The primary data files have %d different sensor-features" % len(feature_names);
print "X is the feature matrix. Each row is an example and each column is a sensor-feature:";
X.shape
print "The primary data files have %s context-labels" % len(label_names);
print "Y is the binary label-matrix. Each row represents an example and each column represents a label.";
print "Value of 1 indicates the label is relevant for the example:";
Y.shape
print "Y is accompanied by the missing-label-matrix, M.";
print "Value of 1 indicates that it is best to consider an entry (example-label pair) as 'missing':";
M.shape



#%%

        # ----------------------------- Set Feature Names -------------------------------------


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names]);
    for (fi,feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc';
            pass;
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro';
            pass;
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet';
            pass;
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc';
            pass;
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass';
            pass;
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc';
            pass;
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud';
            pass;
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP';
            pass;
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS';
            pass;
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF';
            pass;
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat);

        pass;

    return feat_sensor_names;    

feat_sensor_names = get_sensor_names_from_features(feature_names);

for (fi,feature) in enumerate(feature_names):
    print("%3d) %s %s" % (fi,feat_sensor_names[fi].ljust(10),feature));
    pass;
#feat_sensor_names = feat_sensor_names[155:183]
feat_sensor_names = feat_sensor_names[155:181]
print feat_sensor_names


#%%
    
        # ----------------------------- Train Models -------------------------------------

import sklearn.linear_model;

from sklearn import svm, neighbors, metrics, cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, silhouette_score
from sklearn.cluster import KMeans, DBSCAN

def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor);
        use_feature = np.logical_or(use_feature,is_from_sensor);
        pass;
    X = X[:,use_feature];
    return X;

def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train,axis=0);
    std_vec = np.nanstd(X_train,axis=0);
    return (mean_vec,std_vec);

def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
    X_standard = X_centralized / normalizers;
    return X_standard;

def train_model(X_train,Y_train,M_train,feat_sensor_names,label_names,sensors_to_use,target_label):
    # Project the feature matrix to the features from the desired sensors:
    X_train = project_features_to_selected_sensors(X_train,feat_sensor_names,sensors_to_use);
    print("== Projected the features to %d features from the sensors: %s" % (X_train.shape[1],', '.join(sensors_to_use)));

    # It is recommended to standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    (mean_vec,std_vec) = estimate_standardization_params(X_train);
    X_train = standardize_features(X_train,mean_vec,std_vec);
    
    # The single target label:      
    label_ind = label_names.index(target_label);
    y = Y_train[:,label_ind];
    missing_label = M_train[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_train = X_train[existing_label,:];
    y = y[existing_label];
    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized, this is equivalent to assuming average value).
    # You can also further select examples - only those that have values for all the features.
    # For this tutorial, let's use the simple heuristic of zero-imputation:
    X_train[np.isnan(X_train)] = 0.;
    print("== Training with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Now, we have the input features and the ground truth for the output label.
    # We can train a logistic regression model.
    
    # Typically, the data is highly imbalanced, with many more negative examples;
    # To avoid a trivial classifier (one that always declares 'no'), it is important to counter-balance the pos/neg classes:
    #lr_model = sklearn.linear_model.LogisticRegression(class_weight='balanced');

    #Linear SVM
    lr_model = sklearn.svm.LinearSVC(random_state=0)
    #sklearn.svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     #intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     #multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     #verbose=0)
    
    #RF
    #clf = ExtraTreesClassifier(n_estimators=100)
    #lr_model = RandomForestClassifier(n_estimators=185)
    #clf = AdaBoostClassifier(n_estimators=185)
    #lr_model = KNeighborsClassifier(n_neighbors=3)
    #clf = GaussianNB()
    #clf = DecisionTreeClassifier()
    #clf = SVC()
    
    lr_model.fit(X_train,y);
    
    # Assemble all the parts of the model:
    model = {\
            'sensors_to_use':sensors_to_use,\
            'target_label':target_label,\
            'mean_vec':mean_vec,\
            'std_vec':std_vec,\
            'lr_model':lr_model,\
            'X_train':X_train,\
            'y':y};
    
    return model;

target_label = 'TOILET';
Z = X[:,155:181]
#Z = X[:,155:183]
#sensors_to_use = ['Aud', 'AP']
sensors_to_use = ['Aud']
#label_names = ['STAIRS_-_GOING_UP','COOKING','BATHING_-_SHOWER']
#feat_sensor_names = feat_sensor_names[155]'STAIRS_-_GOING_UP'  BATHING_-_SHOWER COOKING

print target_label
model = train_model(Z,Y,M,feat_sensor_names,label_names,sensors_to_use,target_label);


#%%
        # ----------------------------- Test Models for the Same Subject-------------------------------------


def test_model(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names,model):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
    X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,model['sensors_to_use']);
    print("== Projected the features to %d features from the sensors: %s" % (X_test.shape[1],', '.join(model['sensors_to_use'])));

    # We should standardize the features the same way the train data was standardized:
    X_test = standardize_features(X_test,model['mean_vec'],model['std_vec']);
    
    # The single target label:
    label_ind = label_names.index(model['target_label']);
    y = Y_test[:,label_ind];
    missing_label = M_test[:,label_ind];
    existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
    X_test = X_test[existing_label,:];
    y = y[existing_label];
    timestamps = timestamps[existing_label];

    # Do the same treatment for missing features as done to the training data:
    X_test[np.isnan(X_test)] = 0.;
    
    print("== Testing with %d examples. For label '%s' we have %d positive and %d negative examples." % \
          (len(y),get_label_pretty_name(target_label),sum(y),sum(np.logical_not(y))) );
    
    # Preform the prediction:
    y_pred = model['lr_model'].predict(X_test);
    
    # Naive accuracy (correct classification rate):
    accuracy = np.mean(y_pred == y);
    
    # Count occorrences of true-positive, true-negative, false-positive, and false-negative:
    tp = np.sum(np.logical_and(y_pred,y));
    tn = np.sum(np.logical_and(np.logical_not(y_pred),np.logical_not(y)));
    fp = np.sum(np.logical_and(y_pred,np.logical_not(y)));
    fn = np.sum(np.logical_and(np.logical_not(y_pred),y));
    print 'y_pred.shape,tp,tn,fp,fn', tp,tn,fp,fn,y_pred.shape
    
    # Sensitivity (=recall=true positive rate) and Specificity (=true negative rate):
    sensitivity = float(tp) / (tp+fn);
    specificity = float(tn) / (tn+fp);
    
    # Balanced accuracy is a more fair replacement for the naive accuracy:
    balanced_accuracy = (sensitivity + specificity) / 2.;
    
    # Precision:
    # Beware from this metric, since it may be too sensitive to rare labels.
    # In the ExtraSensory Dataset, there is large skew among the positive and negative classes,
    # and for each label the pos/neg ratio is different.
    # This can cause undesirable and misleading results when averaging precision across different labels.
    precision = float(tp) / (tp+fp)
    recall = float(tp) / (tp+fn)
    
    print("-"*10);
    print('Accuracy*:         %.2f' % accuracy);
    print('Sensitivity (TPR): %.2f' % sensitivity);
    print('Specificity (TNR): %.2f' % specificity);
    print('Balanced accuracy: %.2f' % balanced_accuracy);
    print('Precision**:       %.2f' % precision);
    print('Recall**:       %.2f' % recall);
    print("-"*10);
    
    print('* The accuracy metric is misleading - it is dominated by the negative examples (typically there are many more negatives).')
    print('** Precision is very sensitive to rare labels. It can cause misleading results when averaging precision over different labels.')
    
    fig = plt.figure(figsize=(10,4),facecolor='white');
    ax = plt.subplot(1,1,1);
    ax.plot(timestamps[y],1.4*np.ones(sum(y)),'|g',markersize=10,label='ground truth');
    ax.plot(timestamps[y_pred],np.ones(sum(y_pred)),'|b',markersize=10,label='prediction');
    
    seconds_in_day = (60*60*24);
    tick_seconds = range(timestamps[0],timestamps[-1],seconds_in_day);
    tick_labels = (np.array(tick_seconds - timestamps[0]).astype(float) / float(seconds_in_day)).astype(int);
    
    ax.set_ylim([0.5,5]);
    ax.set_xticks(tick_seconds);
    ax.set_xticklabels(tick_labels);
    plt.xlabel('days of participation',fontsize=14);
    ax.legend(loc='best');
    plt.title('%s\nGround truth vs. predicted' % get_label_pretty_name(model['target_label']));
    
    return y_pred;
    
pred1 = test_model(Z,Y,M,timestamps,feat_sensor_names,label_names,model);


#%%    
        # ----------------------------- Test Models for Different Subject -------------------------------------


def validate_column_names_are_consistent(old_column_names,new_column_names):
    if len(old_column_names) != len(new_column_names):
        raise ValueError("!!! Inconsistent number of columns.");
        
    for ci in range(len(old_column_names)):
        if old_column_names[ci] != new_column_names[ci]:
            raise ValueError("!!! Inconsistent column %d) %s != %s" % (ci,old_column_names[ci],new_column_names[ci]));
        pass;
    return;

uuid = 'F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/1155FF54-63D3-4AB2-9863-8385D0BD0A13';
(X2,Y2,M2,timestamps2,feature_names2,label_names2) = read_user_data(uuid);

        
# All the user data files should have the exact same columns. We can validate it:
#validate_column_names_are_consistent(feature_names,feature_names2);
#validate_column_names_are_consistent(label_names,label_names2);

#Z2 = X2[:,155:183]
Z2 = X2[:,155:181]
print Y2.shape
pred2 = test_model(Z2,Y2,M2,timestamps2,feat_sensor_names,label_names,model);

#%%