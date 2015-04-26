################################################################################################
################################################################################################
######################    Title: Sentiment Analysis Project   #########################
########################  Author: Shatrunjai Singh #####################################
########################  Date: 09/20/2014   ###############################################
################################################################################################


#controlling option:
#1, whether include AT_USER and URL in stopwords
#2, Domain specific Stop words


#Unsolved issue
#scaling problem (different dictionaries, different variables) and model issue



import os
import csv
import re
import numpy
import gc
from svm import *
from svmutil import *
from itertools import chain #change two dimensiaonl to one dimensional
import string
import sys
from time import time
import pickle
from collections import Counter
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import numpy as np

#chaning working directory 
os.getcwd()
os.chdir('F:\\Analytics Group\\Jai\\Word List')



###Pre-Processing
#Slang Transform
with open('slang.csv', 'rt') as f:
    slang = csv.reader(f)
    next(slang)
    slang_dict={}
    for row in slang:
        slang_dict.update({row[0]:row[1]})

def slang_transform(tweet):
    for i in slang_dict:
        if i in tweet:
            tweet=tweet.replace(i, slang_dict[i])
    return tweet

#Curse mark Transform
with open('Curse_Mark.csv', 'rt') as f:
    Curse_Mark = csv.reader(f)
    next(Curse_Mark)
    Curse_Mark_dict={}
    for row in Curse_Mark:
        Curse_Mark_dict.update({row[0]:row[1]})

def Curse_Mark_transform(tweet):
    for i in Curse_Mark_dict:
        if i in tweet:
            tweet=tweet.replace(i, Curse_Mark_dict[i])
    return tweet


#Slang transform lower the performance
#importing twitter data
#might need to do the encoding
os.chdir('F:\\Analytics Group\\Jai\\data and output')
#os.chdir('F:\\Analytics Group\\Jai\\Word List')

with open('800_tweet_test_unique.csv', 'rt') as f:
    reader = csv.reader(f)
    next(reader)
    tweet_rating_data={}
    for row in reader:
        tweet=row[0]
        #tweet.encode('utf-8')
        tweet=Curse_Mark_transform(tweet)
        #tweet=slang_transform(tweet)
        tweet_rating_data.update({tweet:int(row[2])})



#dictionary merge function
def dict_merge(dict1,dict2):  
    for i in dict1:
        if i in dict2:
            dict1[i].extend(dict2[i])

#def restart_program()
def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)


t_start=time()

##################################################  Sentiment_Aware_Tokenizer  ##########################################################



# This particular element is used in a couple ways, so we define it
# with a name:
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""



# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
     r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

# This is the core tokenizing regex:
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = str(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = str(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = list(map((lambda x : x if emoticon_re.search(x) else x.lower()), words))
        return words

    def tokenize_random_tweet(self):
        """
        If the twitter library is installed and a twitter connection
        can be established, then tokenize a random tweet.
        """
        try:
            import twitter
        except ImportError:
            print ("Apologies. The random tweet functionality requires the Python twitter library: http://code.google.com/p/python-twitter/")
        from random import shuffle
        api = twitter.Api()
        tweets = api.GetPublicTimeline()
        if tweets:
            for tweet in tweets:
                if tweet.user.lang == 'en':            
                    return self.tokenize(tweet.text)
        else:
            raise Exception("Apologies. I couldn't get Twitter to give me a public English-language tweet. Perhaps try again")

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, chr(entnum))	
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = list(filter((lambda x : x != amp), ents))
        for ent in ents:
            entname = ent[1:-1]
            try:            
                s = s.replace(ent, chr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass                    
            s = s.replace(amp, " and ")
        return s

    def tokenize_with_cap(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = str(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = str(s)
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = list(map((lambda x : x if emoticon_re.search(x) else x), words))
        return words


    

if __name__ == '__main__':
    tok = Tokenizer(preserve_case=False)


####Negation tokenizer
#set is faster
negation=set(['never','no','nothing','nowhere','noone','none','not','havent','hasnt','hadnt','cant','couldnt','shouldnt','wont',
          'wouldnt','dont','doesnt','didnt','isnt','arent','aint'])  

clause_punct=(r"""(^[,.:;!?]$)""")
clause_punct=set(clause_punct)

def no_neg(word):
        return word


not_neg=["%","&","(:","!","):",");",")':","*",".",". . . .",". ..",". ."]

def import_dict_with_colname(path):
    with open (path,'rt') as f:
        table=csv.reader(f)
        dict_lexicons={}
        next(table)  # skip the first line
        for row in table:
            dict_lexicons.update({row[0]:row[1]})
    return dict_lexicons

#os.chdir('F:\\Analytics Group\\Jai\\data and output')
os.chdir('F:\\Analytics Group\\Jai\\Word List')

Emoticon_Neg=import_dict_with_colname("Emoticon_Neg.csv")
Emoticon_Pos=import_dict_with_colname("Emoticon_Pos.csv")


def neg(word):
    if (word in not_neg) or (word in Emoticon_Neg) or (word in Emoticon_Pos) or (word in domain_specific_stopwords):
        word=word
    else:
        word=word+"_NEG"
    return word

def toknize_with_neg(tweet):
    words=tok.tokenize(tweet)
    words_with_negation=[]
    neg_device=no_neg
    for i in range(len(words)):
        if words[i] in negation or words[i].endswith(r"n't"):      
            neg_device=neg
            words_with_negation.append(words[i])
        elif words[i] in clause_punct:
            neg_device=no_neg
            words_with_negation.append(words[i])
        else:
            word=neg_device(words[i])
            words_with_negation.append(word)
    return words_with_negation
    
#sys.exit()

#############################################Unigram Vector##############################################################

t_Unigram_start=time()

#stopwords, either import the AT_USER and URL or not 

with open('stopwords.csv', 'rt') as s:
    reader = csv.reader(s)
    stopwords  =[]
    for row in reader:
        stopwords.append(row)
        
stopwords=list(chain.from_iterable(stopwords))

stopwords.append('at_user')
stopwords.append('urls')
stopwords.append('rt')
stopwords.append('at_user_neg')
stopwords.append('urls_neg')
stopwords.append('rt_neg')

stopwords=set(stopwords)

domain_specific_stopwords={"discover","chase","citi","citibank","america","american","express","americanexpress","citibank","fifth","third","pnc","wells","fargo","wellsfargo","capital","one"}

#whether to combine domain_specific_stopwords or not
stopwords=stopwords.union(domain_specific_stopwords)


#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#Replace URL links
def replaceURL(s):
    pattern = re.compile(r"((www\.[\s]+)|(https?://[^\s]+))", re.DOTALL)
    return pattern.sub("urls", s)
#end



#start getfeatureVector/ optional with # and http
def getFeatureVector(tweet):
    featureset = []
    #Convert www.* or https?://* to URL
    tweet=replaceURL(tweet)
    #tokenize the tweet
    words = toknize_with_neg(tweet)
    #words = tok.tokenize(tweet)
    for w in words:
        #whether to group the number/date/phone number
        number_pattern=re.compile(r"""(?:[+\-]?\d+[,/.:-]\d+[+\-]?)|(^\d+$)""")
        if number_pattern.match(w):
            w="number_date_time"
        #Convert @username to AT_USER
        w=  re.sub(r'@([^\s]+)', 'at_user', w)
        #Replace #word with word
        w = re.sub(r'#([^\s]+)', r'\1', w)
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        w = w.lower()
        #ignore if it is a stop word
        if w not in stopwords:
            featureset.append(w)
        else:
            continue
    return featureset
#end




###Unigram Feature Building
#Using Dictionary---key is unique, get rid of duplicate tweets
Raw_featureList=[]
Uni_featureList=set()
Uni_feature={}



for tweet in tweet_rating_data:
    feature_set = getFeatureVector(tweet)
    Raw_featureList.extend(feature_set)
    Uni_featureList=Uni_featureList.union(set(feature_set))
    Uni_feature.update({tweet:feature_set})

#check the most n most frequency feature
#Counter(Raw_featureList).most_common(100)



# Unigram Feature Selection
# For the purpose of computation efficiency and perhaps accuracy improvement

Feature_selection_list=set()

for i in Uni_featureList:
    if Raw_featureList.count(i)>1:
        Feature_selection_list.add(i)

#whether to turn on the frequency feature selection
#Uni_featureList=Feature_selection_list

#print(time()-t_start)

#if using list: Uni_featureList=list(chain.from_iterable(Uni_featureList))  ---chain.from_iterable iterate list inside the list while chain just iterate inside the list
#sys.exit()

#write list to csv file
"""
with open('feature_set.csv', 'w', newline='') as csvfile:
	listwriter = csv.writer(csvfile)
	for i in a:
		listwriter.writerow([i])
"""


###Unigram vector Building
UnigramVector={}
for tweet in Uni_feature:
    Uni_vector={}                   # leverage dictionary to build the vector
    for feat in Uni_featureList:        # Initialize the dictionary
        Uni_vector[feat]=0
    for word in Uni_feature[tweet]:            # fill the dictionary
        if word in Uni_vector:
            Uni_vector[word]=1
    vector=list(Uni_vector.values())# turn the dictionary value to list
    UnigramVector.update({tweet:vector})

Unigram_Dict=UnigramVector
Unigram_Keys=Uni_vector.keys()


with open('Unigram_dict.csv', 'w', newline='') as csvfile:
	listwriter = csv.writer(csvfile)
	for i in Unigram_Keys:
		listwriter.writerow([i])

#end



t_Unigram_end=time()

print ("Unigram Vector Runing Time")
print (t_Unigram_end-t_Unigram_start)

#restart_program()
#pickle.dump(Unigram_Vector,open("Unigram_Vector.p","wb"))
#check point
#sys.exit()

#############################################Character N Gram Vector##################################################


#character ngrams no punctuation, lower case
def ngrams_char(tweet,n):
    tweet=tweet.lower()
    tweet=re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    sequence=list(tweet)
    count=max(0,len(sequence)-n+1)
    return [tuple(sequence[i:i+n]) for i in range(count)]



##################################################  Lexicons  ##########################################################

t_Lexicons_start=time()

def import_dict_with_colname(path):
    with open (path,'rt') as f:
        table=csv.reader(f)
        dict_lexicons={}
        next(table)  # skip the first line
        for row in table:
            dict_lexicons.update({row[0]:row[1]})
    return dict_lexicons


Bad_Word_Orig=import_dict_with_colname("Bad_Word_Orig.csv")

BoosterWord_Neg=import_dict_with_colname("BoosterWord_Neg.csv")
BoosterWord_Pos=import_dict_with_colname("BoosterWord_Pos.csv")

Combined_Neg=import_dict_with_colname("Combined_Neg.csv")
Combined_Pos=import_dict_with_colname("Combined_Pos.csv")

Emotional_Orig_Neg=import_dict_with_colname("Emotional_Orig_Neg.csv")
Emotional_Orig_Pos=import_dict_with_colname("Emotional_Orig_Pos.csv")

Negating_Word=import_dict_with_colname("Negating_Word.csv")

Negative_Word=import_dict_with_colname("Negative_Word.csv")
Positive_Word=import_dict_with_colname("Positive_Word.csv")

#DAL_Plea_Pos_Word=import_dict_with_colname("DAL_Plea_Pos.csv")
#DAL_Plea_Neg_Word=import_dict_with_colname("DAL_Plea_Neg.csv")


#white space tokenizer for lexicon approach
#http://stackoverflow.com/questions/12705293/regex-to-split-words-in-python


#does not fix the "'we are good', how about you?' problem!!!
def white_space_tokenize(tweet):
    """

    :param tweet:
    :return:
    """
    to_be_removed=["!","#",".","?","@",";",",",":",'"']
    to_be_removed=set(to_be_removed)
    tweet=replaceURL(tweet)
    for p in to_be_removed:
        tweet=tweet.replace(p," ")
    wordlist=tweet.split()
    for i in range(len(wordlist)):
        wordlist[i] = replaceTwoOrMore(wordlist[i])
        wordlist[i]=wordlist[i].lower()
        if (wordlist[i][0]!=r"'"):
                wordlist[i]=wordlist[i]
        else:
                wordlist[i]=wordlist[i][1:]
    return wordlist

def white_space_toknize_with_neg(tweet):
    words=white_space_tokenize(tweet)
    words_with_negation=[]
    neg_device=no_neg
    for i in range(len(words)):
        if words[i] in negation or words[i].endswith(r"n't"):      
            neg_device=neg
            words_with_negation.append(words[i])
        elif words[i] in clause_punct:
            neg_device=no_neg
            words_with_negation.append(words[i])
        else:
            word=neg_device(words[i])
            words_with_negation.append(word)
    return words_with_negation




#need to figure it out how to do the score of the last token in the tweet with score(w,p)>0
def Lexicon_Score(tweet_rating_data,dictionary):
    data={}    
    for tweet in tweet_rating_data:
        wordlist=set(white_space_tokenize(tweet))
        sum_score=0
        max_score_positive=0
        max_score_negative=0
        total_count_tokens_positive=0
        total_count_tokens_negative=0
        last_token_score=0
        for word in wordlist:
            if word in dictionary:
                score=int(dictionary[word])  #if use DAL, then need to change to Float
                sum_score=sum_score+score
                if score>=0:
                    total_count_tokens_positive=total_count_tokens_positive+1
                    if score>max_score_positive:
                        max_score_positive=score
                elif score<0:
                    total_count_tokens_negative=total_count_tokens_negative+1
                    if score<total_count_tokens_negative:
                        max_score_negative=score
        data.update({tweet:[sum_score,max_score_positive,max_score_negative,total_count_tokens_positive,total_count_tokens_negative,last_token_score]})                
    return data




###Lexicons Feature###  #time consuming
def getLexicons_Vector(tweet_rating_data):
    #1
    Bad_Word_Orig_score=Lexicon_Score(tweet_rating_data,Bad_Word_Orig)
    #2
    BoosterWord_Neg_score=Lexicon_Score(tweet_rating_data,BoosterWord_Neg)
    BoosterWord_Pos_score=Lexicon_Score(tweet_rating_data,BoosterWord_Pos)
    #3
    Combined_Neg_score=Lexicon_Score(tweet_rating_data,Combined_Neg)
    Combined_Pos_score=Lexicon_Score(tweet_rating_data,Combined_Pos)
    #4
    Emotional_Orig_Neg_score=Lexicon_Score(tweet_rating_data,Emotional_Orig_Neg)
    Emotional_Orig_Pos_score=Lexicon_Score(tweet_rating_data,Emotional_Orig_Pos)
    #5
    Negating_Word_score=Lexicon_Score(tweet_rating_data,Negating_Word)
    #6
    Negative_Word_score=Lexicon_Score(tweet_rating_data,Negative_Word)
    Positive_Word_score=Lexicon_Score(tweet_rating_data,Positive_Word)
    #7  DAL not helpful
    #DAL_Plea_Pos_Word_score=Lexicon_Score(tweet_rating_data,DAL_Plea_Pos_Word)
    #DAL_Plea_Neg_Word_score=Lexicon_Score(tweet_rating_data,DAL_Plea_Neg_Word)
    #combine vector
    Lexicons_vector=Bad_Word_Orig_score
    dict_merge(Lexicons_vector,BoosterWord_Neg_score)
    dict_merge(Lexicons_vector,BoosterWord_Pos_score)
    dict_merge(Lexicons_vector,Combined_Neg_score)
    dict_merge(Lexicons_vector,Combined_Pos_score)
    dict_merge(Lexicons_vector,Emotional_Orig_Neg_score)
    dict_merge(Lexicons_vector,Emotional_Orig_Pos_score)
    dict_merge(Lexicons_vector,Negating_Word_score)
    dict_merge(Lexicons_vector,Negative_Word_score)
    dict_merge(Lexicons_vector,Positive_Word_score)
    #list_merge(Lexicons_vector,DAL_Plea_Pos_Word_score)
    #list_merge(Lexicons_vector,DAL_Plea_Neg_Word_score)
    return Lexicons_vector

Lexicons_Dict=getLexicons_Vector(tweet_rating_data)


t_Lexicons_end=time()

print ("Lexicons Vector Runing Time")
print (t_Lexicons_end-t_Lexicons_start)

#pickle.dump(Lexicons_Dict,open("Lexicons_Dict.p","wb"))
#restart_program()


#checking point
#sys.exit()

################################################  Encodings  ########################################################
##########################################Input and output is vector#################################################

t_Encoding_start=time()

#encoding tokenizer, mainly for All_caps
def encoding_tokenizer(tweet):
    vector=set()
    #Convert www.* or https?://* to URL
    tweet=replaceURL(tweet)
    #tokenize the tweet with cap
    words = set(tok.tokenize_with_cap(tweet))
    for w in words:
        #Convert @username to AT_USER
        w=  re.sub(r'@([^\s]+)', 'at_user', w)
        #Replace #word with word
        w = re.sub(r'#([^\s]+)', r'\1', w)
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #ignore if it is a stop word
        if w.lower() in stopwords:
            continue
        else:
            vector.add(w)
    return vector


#Number of Negation#
def n_Negation(tweet):
    #Convert www.* or https?://* to URL
    tweet=replaceURL(tweet)
    wordlist=toknize_with_neg(tweet)
    #wordlist=tok.tokenize(tweet)
    reps=0
    for i in range(len(wordlist)):
        if wordlist[i].endswith(r"_NEG"):
            count=count+1
            if count==1:
                reps=reps+1
        else:
            count=0
    return reps


#encoding all in one function:
def encoding_vector(tweet_rating_data):
    encoding_vectors={}
    for tweet in tweet_rating_data:
        encoding_vector= encoding_tokenizer(tweet)
        #punctuation
        tweet_n_exclamation=tweet.count("!")
        tweet_n_question=tweet.count("?")
        n_hashtag=tweet.count("#")
        #all caps and elongated
        boolean_allcaps=[]
        bool_list_elongated=[]
        for word in encoding_vector:
            boolean_allcaps.append(word.isupper())
            bool_list_elongated.append(bool(re.search(r'(.)\1\1+',word)))
        tweet_num_caps=int(numpy.sum(boolean_allcaps))
        tweet_num_elongated=int(numpy.sum(bool_list_elongated))
        #last one is ? or !
        whe_punct=0
        if "!" in tweet[-1] or "?" in tweet[-1]:
            whe_punct=1
        #continuous ! or ?
        n_conti_excl_ques=0
        count=0
        for i in tweet:
            if "!" in i or "?" in i:
                count=count+1
                if count==2:
                    n_conti_excl_ques=n_conti_excl_ques+1
                else:
                    count=0
        #negation      
        num_negation=n_Negation(tweet)

        encoding_vectors.update({tweet:[tweet_n_exclamation,tweet_n_question,n_hashtag,tweet_num_caps,tweet_num_elongated,whe_punct,n_conti_excl_ques,num_negation]})
    return encoding_vectors


#emticons#
Emoticon_Neg=import_dict_with_colname("Emoticon_Neg.csv")
Emoticon_Pos=import_dict_with_colname("Emoticon_Pos.csv")

def emoticon(tweet_rating_data,pos_dictionary,neg_dictionary):
    emoticon_vector={}
    for tweet in tweet_rating_data:
        wordlist=tok.tokenize(tweet)
        last_token_pos=0
        last_token_neg=0
        pos_score=0
        neg_score=0
        for i in range(len(wordlist)):
            if wordlist[i] in pos_dictionary:
                pos_score=1
            elif wordlist[i] in neg_dictionary:
                neg_score=1
        if wordlist[-1] in pos_dictionary:
            last_token_pos=1
        if wordlist[-1] in neg_dictionary:
            last_token_pos=1

        emoticon_vector.update({tweet:[pos_score,last_token_pos,neg_score,last_token_neg]})
    return emoticon_vector

###Encoding Vector Building###
encoding_dict=encoding_vector(tweet_rating_data)
emoticon_dict=emoticon(tweet_rating_data,Emoticon_Pos,Emoticon_Neg)


t_Encoding_end=time()

print ("Encoding Vector Runing Time")
print (t_Encoding_end-t_Encoding_start)


#pickle.dump(emoticon_dict,open("emoticon_dict.p","wb"))
#restart_program()
#sys.exit()


############################  SVM Feature Vector Building and Model Statistics and Evaluation #########################################


t_model_building_start=time()

Model_Predictor=[]


# Unigram---968 features
dict_merge(Unigram_Dict,Lexicons_Dict)
#del Lexicons_Dict---60 features
dict_merge(Unigram_Dict,encoding_dict)
#del encoding_dict---8  features  emoticon---4
dict_merge(Unigram_Dict,emoticon_dict)

tweet_text=list(Unigram_Dict.keys())

#dirty way to order the dictionary
for i in tweet_text:
    if i in tweet_rating_data:
        Model_Predictor.append(int(tweet_rating_data[i]))


Model_Feature=list(Unigram_Dict.values())

gc.collect()

t_model_building_ending=time()
print ("Vector Combining Runing Time")
print (t_model_building_ending-t_model_building_start)

### SVM Model Statistics and Evaluation  
t_SVM_start=time()


#0.5 is for 10 thousand
#1 for 800

X_train, X_test, y_train, y_test = train_test_split(Model_Feature, Model_Predictor, test_size=0.2,random_state=1989)

sys.exit()
# svm classification
clf = svm.SVC(kernel='linear', C = 1.0).fit(X_train, y_train)
y_predicted = clf.predict(X_test)
cv=cross_validation.cross_val_score(clf,X_train,y_train,cv=10)
# performance
print ("cross validation result(10 fold)")
print (np.mean(cv))
print ("Classification report for %s" % clf)
print (metrics.classification_report(y_test, y_predicted))
print ("Confusion matrix")
print (metrics.confusion_matrix(y_test, y_predicted))


sys.exit()







"""
sys.exit()


model=svm_train(Model_Predictor,Model_Feature,'-s 0 -t 0 -c 2')



#save model
os.chdir('F:\\Analytics Group\\Jai\\data and output')
#os.chdir('F:\\Analytics Group\\Jai\\Word List')
svm_save_model("libsvm.model",model)




#cross validation--- return cost, and accuracy
def cv_cost(upper_boundary, lower_boundary, tolerance):
    u=upper_boundary
    l=lower_boundary
    accuracy_u=svm_train(Model_Predictor,Model_Feature,'-v 10 -s 0 -t 0 -c %s' %u) 
    accuracy_l=svm_train(Model_Predictor,Model_Feature,'-v 10 -s 0 -t 0 -c %s' %l)
    diff=upper_boundary-lower_boundary
    mid=(u+l)/2
    while diff>tolerance:
        mid=(u+l)/2
        if accuracy_u>accuracy_l:
            l=mid
            print (l)
            accuracy_l=svm_train(Model_Predictor,Model_Feature,'-v 10 -s 0 -t 0 -c %s' %l)
            accuracy=accuracy_l
        elif accuracy_u<accuracy_l:
            u=mid
            print (u)
            accuracy_u=svm_train(Model_Predictor,Model_Feature,'-v 10 -s 0 -t 0 -c %s' %u)
            accuracy=accuracy_u
        diff=u-l
    return [mid, accuracy]

print ("cross validation result")    
cv_cost(10, 0.1, 0.1)

t_SVM_end=time()

print ("SVM Model Building Runing Time")
print (t_SVM_end-t_Encoding_start)



t_end=time()
print ("Total Run Time")
print (t_end-t_start)

"""
sys.exit()

############# Model Evaluation ############
#http://stackoverflow.com/questions/16927964/how-to-calculate-precision-recall-and-f-score-with-libsvm-in-python


"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# svm classification
clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0).fit(X_train, y_train)
y_predicted = clf.predict(X_test)

# performance
print "Classification report for %s" % clf
print
print metrics.classification_report(y_test, y_predicted)
print
print "Confusion matrix"
print metrics.confusion_matrix(y_test, y_predicted)




p_labels, p_accs, p_vals = svm_predict(Model_Predictor,Model_Feature, accuracy)
"""

################## Naive Bayes ############

from sklearn.naive_bayes import MultinomialNB

for m in range(len(Model_Feature)):
	for n in range(len(Model_Feature[m])):
		Model_Feature[m][n]=abs(Model_Feature[m][n])

NB = MultinomialNB()

NB.fit(Model_Feature, Model_Predictor)

print (NB.score(Model_Feature, Model_Predictor))


from sklearn import svm
Support_VM=svm.LinearSVC(Model_Feature,Model_Predictor)
