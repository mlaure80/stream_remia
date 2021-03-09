
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


# ml
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,f1_score

# adding pre-processing to scale data which is inherent in the Pipeline in the pyspark done earlier
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# saving model library
import pickle

def check_words(sentence,words):
  """
  input_1: text ie. sentence
  input_2: list of words to check, and count the occurrence
  returns: a count of appearances

  """
  count_w=0
  for w in words:
    if sentence.find("not " + w.lower()) > -1:
      # print("not " + w.lower())
      # do nothing; do not increment counter as this exists
      pass
    else:
      # print("not " + w.lower())
      count_w+=sentence.count(w.lower())
      # pass
  return count_w

# create statistics on my model
def createDF_stats(orig,pred):
    """[creates dataframw with model results]

    Args:
        orig ([type]): [y_test]
        pred ([type]): [y_predict]
    """
    df_eval=pd.DataFrame({'y_test':orig.ravel(),'y_pred':pred})
    df_eval['match']=df_eval['y_test']==df_eval['y_pred']
    return df_eval

def load_data():
    # assume data file will always be the same per training
    data = pickle.load(open('./df.pkl', 'rb'))
    return data

def load_model(model_file_name):
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    return loaded_model


def runModel():
    """
    this function has model logic that works on data
    """

    good_words=['tasty','flavorful','love','great','nice','good']
    bad_words=['bad','terrible','nasty','horrible','awful','disgusting','not']
    
    #data
    url = "https://s3.amazonaws.com//dataviz-curriculum/day_3/ratings_and_sentiments.csv"

    import_fields= ['coffee_shop_name',
                    'review_text',
                    'num_rating',
                    'bool_HIGH']

    df_rating=pd.read_csv(url,encoding='unicode_escape',usecols=import_fields)
    df_rating['review_text']=df_rating['review_text'].str.lower()

    df_rating['words']=df_rating['review_text'].str.split()

    df_rating['n_words']=df_rating['words'].apply(lambda x:len(x))
    df_rating['good_words']=df_rating['review_text'].apply(lambda sentence:check_words(sentence,good_words))
    df_rating['bad_words']=df_rating['review_text'].apply(lambda sentence:check_words(sentence,bad_words))

    df_rating['label']=pd.to_numeric(df_rating['num_rating'],downcast='integer')

    df_rating['review_date']=df_rating['review_text'].str.split(" ",expand=True).iloc[:,0:1]

    # features listed here
    features=['n_words','good_words','bad_words']

    # silo the dataframe for later access
    all_columns=['review_date','coffee_shop_name','review_text','words',
                'n_words','good_words','bad_words','label','bool_HIGH']

    df_rating[all_columns].to_pickle("./df.pkl")

    X=df_rating[features]
    y=df_rating['label']

    # try with X_scaled
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    gnb = GaussianNB()
    # gnb.fit(X_train,y_train)
    # gnb.predict(X_test)
    y_pred=gnb.fit(X_train,y_train).predict(X_test)

    model_file='./models/gnb.sav'
    # dumping or serializing an object to a file in binary mode

    # file_obj=open(model_file,'wb')

    pickle.dump(gnb,open(model_file,'wb'))


    pickle.dump()

if  __name__ =='__main__':
    runModel()
