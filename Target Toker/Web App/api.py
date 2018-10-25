import numpy as np
import pickle

pipeline = pickle.load(open('./model.pkl', 'rb'))


def make_prediction(features):
    X = np.array([int(features['Health'] == 'Yes'),
                 int(features['Enrolled'] == 'Yes'),
                 int(features['Ethnicity1'] == 'Yes'),
                 int(features['Income'] == 'Yes'),
                 int(features['Education1'] == 'Yes'),
                 int(features['Marital_Status1'] == 'Yes'),
                 int(features['Age'] == 'Yes'),
                 int(features['Ethnicity2'] == 'Yes'),
                 int(features['Marital_Status2'] == 'Yes'),
                 int(features['Education2'] == 'Yes')]).reshape(1,-1)

    prob_target_toker = pipeline.predict_proba(X)[0, 1]

    result = {
        'prediction': int(prob_target_toker > 0.5),
        'prob_target_toker': "{0:.2f}".format(prob_target_toker)
    }
    return result
