import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection as skms
import sklearn.naive_bayes as NB

def read_data(path1:str, path2:str) -> np.ndarray:
    """ Read in crab dataset and split into labels/features
    
    Args:
        path1(str): location of dataset 1
        path2(str): location of dataset 2

    Returns:
        np.ndarray: numpy labels array, features array
    
    """
    scdf = pd.read_csv(path1)
    stcdf = pd.read_csv(path2)
    crab = pd.concat([scdf,stcdf]) #combine crab datasets

    le = sk.preprocessing.LabelEncoder()
    crab2 = crab.apply(le.fit_transform) #encode values to numerical
    labels = np.asarray(crab2['name']) #label we seek to predict (crab species)
    #note: snow crab = 0, southern tanner crab = 1
    
    #location, depth, and water temp features used for prediction
    feats = np.asarray(crab2[['latitude', 'longitude', 'bottom_depth', 'surface_temperature', 'bottom_temperature']])
    return labels, feats

def train(train_data: np.ndarray, train_labels: np.ndarray) -> NB.GaussianNB:
    """ Train Naive-Bayes binary classifier based on training features and labels

    Args:
        train_data (np.ndarray): numpy array with training features
        train_labels (np.ndarray): numpy array with training labels

    Returns:
        NB.GaussianNB: fitted model

    """
    model = NB.GaussianNB()
    model.fit(train_data, train_labels)
    return model

def predict(model: NB.GaussianNB, test_data: np.ndarray) -> np.ndarray:
    """ Predict labels from test data

    Args:
        model (NB.GaussianNB): previously trained model
        test_data (np.ndarray): numpy array of test features

    Returns:
        model_labels (np.ndarray): numpy array of predicted labels

    """
    model_labels = model.predict(test_data)
    return model_labels

def assess(model_labels: np.ndarray, test_labels: np.ndarray) -> tuple():
    """ Assess binary model

    Args:
        model_labels (np.ndarray): numpy array of predicted labels from model
        test_labels (np.ndarray): numpy array of test labels

    Returns:
        tuple(): tuple containing floats: accuracy, precision, recall, and F1 score of model
        
    """
    true_all = np.where(model_labels==test_labels)[0]
    true_pos = np.where(test_labels[true_all]==0)[0]
    false_all = np.where(model_labels!=test_labels)[0]
    false_pos = np.where(test_labels[false_all]==1)[0]
    false_neg = np.where(test_labels[false_all]==0)[0]
    
    accuracy = len(true_all)/len(test_labels)
    precision = len(true_pos)/(len(true_pos) + len(false_pos))
    recall = len(true_pos)/(len(true_pos) + len(false_neg))
    f1score = (2*precision*recall)/(precision+recall)
  
    print('Accuracy:  ', f"{accuracy:.1%}")
    print('Precision: ', f"{precision:.1%}")
    print('Recall:    ', f"{recall:.1%}")
    print('F1 Score:  ', f"{f1score:.1%}")
    return (accuracy, precision, recall, f1score)

if __name__ == '__main__':
    """ 🦀 Test on crab species example: snow crabs and southern tanner crabs 🦀 """

    crab_labels, crab_data = read_data('mfsnowcrab.csv', 'southerntannercrab.csv') #read in data, split into labels/features

    train_crab_data, test_crab_data, train_crab_labels, test_crab_labels = skms.train_test_split(crab_data, 
        crab_labels, test_size=.25) #train/test split crab data
    
    crab_model = train(train_crab_data, train_crab_labels) #train model on training crab data
    crab_model_labels = predict(crab_model, test_crab_data) #predict crab species using trained model
    crab_assess = assess(crab_model_labels, test_crab_labels) #assess model performance

    print("\nOut of all crabs predicted to be snow crabs by our model, ", f"{crab_assess[1]:.1%}", " were correct.") #precision
    print("Out of all snow crabs in our test data, ", f"{crab_assess[2]:.1%}", " were correctly classfied.") #recall
    print("Overall, out of all predictions by our model, ", f"{crab_assess[0]:.1%}", " were correct.") #accuracy
