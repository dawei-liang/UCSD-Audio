# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:15:55 2018

@author: david
"""
    #Get sound data and time frames
    #sample_rate, data = wavfile.read('test.wav')
    #time = np.arange(len(data))/float(sample_rate)
    
    
    #signal = np.fromstring(signal, 'Int16')
    
   # https://gist.github.com/leouieda/9043213
   # https://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
           
   '''本代码后两块与第一块相对独立，只是共用了第一块训练的model。由于第一部分已经定义了主函数，
   所以本程序要分块跑，不然中间定义的函数可能会执行不到'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

import librosa
import soundfile as sf
import python_speech_features

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import numpy as np

import gzip;
import StringIO


#%%
    # ----------------------------- Model training and testing -------------------------------------  
             
if __name__ == "__main__": 
    name = []
    name.append('test.wav')
    name.append('140453__aesqe__cooking-02.wav')
    name.append('364903__tieswijnen__cooking-boiling.wav')
    name.append('42811__kidscasttechy__boiling-liquid.wav')
    name.append('215143__cjwilso23__egg-cooking.wav')
    name.append('352050__kenzievaness__cooking-in-the-kitchen.wav')
    name.append('360786__dutchlady__prei-snijden-hakken.wav')
    name.append('262266__gowlermusic__toilet-flush.wav')
    name.append('340053__iesp__toilet-flushing.wav')
    name.append('340897__passairmangrace__toiletflush-1-bip.wav')
    name.append('400577__inspectorj__bathroom-extractor-fan-a.wav')
    name.append('401744__inspectorj__toilet-flush-english-a.wav')
    name.append('G:/Research1/codes/washinghands/139065__funnyman374__washinghands.wav')
    name.append('G:/Research1/codes/washinghands/234251__rivernile7__washing-hands.wav')
    name.append('G:/Research1/codes/washinghands/257960__fillsoko__washing-hands.wav')
    name.append('G:/Research1/codes/washinghands/335263__mivori__boom-bathroom-washinghands.wav')
    name.append('G:/Research1/codes/washinghands/380758__sempoo__hand-washing.wav')
    name.append('G:/Research1/codes/keyboard/211601__bendthebasics__small-keyboard-clicks.wav')
    name.append('G:/Research1/codes/keyboard/261534__thalamus-lab__clavicordio-resonance-board-hit.wav')
    name.append('G:/Research1/codes/keyboard/326972__wingsofirony__key-board.wav')
    name.append('G:/Research1/codes/keyboard/344220__vsokorelos__macbook-keyboard-sound.wav')
    name.append('G:/Research1/codes/keyboard/391191__spalena__keys-on-plastic-board.wav')
    
    name.append('G:/Research1/codes/recordfromAudioset/toilet.wav')
    name.append('274448__polytest__toilet-flushing.wav')
    name.append('G:/Research1/codes/recordfromAudioset/frying.wav')
    name.append('G:/Research1/codes/recordfromAudioset/keyboard.wav')
    data, samplerate = sf.read(name[0]) 
    mean = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048)).mean(axis=0)
    var = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048)).var(axis=0)
    features = mean
    features = np.hstack((mean, var))
    for i in range (1,len(name) - 4):
        data, samplerate = sf.read(name[i]) 
        temp = python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048)
        mean = temp.mean(axis=0)
        var = temp.var(axis=0)
        temp = np.hstack((mean, var))
        features = np.vstack((features, temp))
    

    y=['t','c','c','c','c','c','c','t','t','t','t','t','w','w','w','w','w',
       'k','k','k','k','k']
    #y=['t','c','c','c','c','c','c','t','t','t','t','t']
#
#    clf=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#     verbose=0)
    #clf = LinearSVC(random_state=0)
    clf = RandomForestClassifier(n_estimators=185)
    clf.fit(features, y)
#

#
#print(clf.coef_)
#print(clf.intercept_)
    
##  test 1(c)
    print name[len(name)-2]
    data, samplerate = sf.read(name[len(name)-2])
    mean = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048)).mean(axis=0)
    var = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048)).var(axis=0)
    testset = mean
    testset = np.hstack((mean, var))
    testset = testset.reshape(1,26)
    print(clf.predict(testset))

##  test 2(c)
    print name[len(name)-1]
    data, samplerate = sf.read(name[len(name)-1])
    mean = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048,winfunc=np.hamming)).mean(axis=0)
    var = (python_speech_features.base.mfcc(data, samplerate=samplerate,
                                            numcep = 13,nfft=2048,winfunc=np.hamming)).var(axis=0)
    testset = mean
    testset = np.hstack((mean, var))
    testset = testset.reshape(1,26)
    print(clf.predict(testset))

#%%
        # ----------------------------- Import UCSD Labels and Features -------------------------------------


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
        # ----------------------------- Test on UCSD dataset -------------------------------------


        
# All the user data files should have the exact same columns. We can validate it:
#validate_column_names_are_consistent(feature_names,feature_names2);
#validate_column_names_are_consistent(label_names,label_names2);
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
    
    
    def project_features_to_selected_sensors(X,feat_sensor_names,sensors_to_use):
        use_feature = np.zeros(len(feat_sensor_names),dtype=bool);
        for sensor in sensors_to_use:
            is_from_sensor = (feat_sensor_names == sensor);
            use_feature = np.logical_or(use_feature,is_from_sensor);
            pass;
            X = X[:,use_feature];
            return X;
    
    def standardize_features(X,mean_vec,std_vec):
    # Subtract the mean, to centralize all features around zero:
        X_centralized = X - mean_vec.reshape((1,-1));
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
        normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1,-1));
        X_standard = X_centralized / normalizers;
        return X_standard;

    
    def Extract_features(X_test,Y_test,M_test,timestamps,feat_sensor_names,label_names):
    # Project the feature matrix to the features from the sensors that the classifier is based on:
        X_test = project_features_to_selected_sensors(X_test,feat_sensor_names,sensors_to_use);

    # We should standardize the features the same way the train data was standardized:
    
    
    # The single target label:
        label_ind = label_names.index('COOKING');
        missing_label = M_test[:,label_ind];
        existing_label = np.logical_not(missing_label);
    
    # Select only the examples that are not missing the target label:
        X_test = X_test[existing_label,:];

    # Do the same treatment for missing features as done to the training data:
        X_test[np.isnan(X_test)] = 0.;
    
    # Preform the prediction:
        return X_test
#Z2 = X2[:,155:183]
        
    uuid = 'F:/audio_database_ucsd/ExtraSensory.per_uuid_features_labels/1155FF54-63D3-4AB2-9863-8385D0BD0A13';
    (X2,Y2,M2,timestamps2,feature_names2,label_names) = read_user_data(uuid);
    Z2 = X2[:,155:181]
    
    feat_sensor_names = get_sensor_names_from_features(feature_names2);
    feat_sensor_names = feat_sensor_names[155:181]
    sensors_to_use = ['Aud']
    
    X_test = Extract_features(Z2, Y2, M2, timestamps2, feat_sensor_names, label_names)
    k = (clf.predict(X_test))
    
