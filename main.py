# Importing libraries.
import pandas as pd
import re
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import src.test as test 


new_summary = '''

Though he battled for years to marry her, Henry VIII has become disenchanted with the audacious Anne Boleyn. She has failed to give him a son, and her sharp intelligence and strong will have alienated his old friends and the noble families of England.

When the discarded Katherine, Henry's first wife, dies in exile from the court, Anne stands starkly exposed, the focus of gossip and malice, setting in motion a dramatic trial of the queen and her suitors for adultery and treason.

At a word from Henry, Thomas Cromwell is ready to bring her down. Over a few terrifying weeks, Anne is ensnared in a web of conspiracy, while the demure Jane Seymour stands waiting her turn for the poisoned wedding ring. But Anne and her powerful family will not yield without a ferocious struggle. To defeat the Boleyns, Cromwell must ally himself with his enemies. What price will he pay for Annie's head?"
'''
    
