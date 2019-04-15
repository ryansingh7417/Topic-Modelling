import pandas as pd
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

df = pd.read_csv("poem.csv")
dt = df.poem[:1000]
dt = dt.values.tolist()

#function to create keywords from a document
def sent_word(sent):
    for sents in sent:
        yield(gensim.utils.simple_preprocess(str(sents), deacc=True))
data = list(sent_word(dt))

#function to remove stopwords
def rm_stopword(texts):
    return([w for w in texts if not w in stop_words])     
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

fldata = []
for i in data:
    fldata.append(rm_stopword(i))
    
#creating the Dictionary and Corpus needed for Topic Modeling
id2word = corpora.Dictionary(fldata)

#Term Document Frequency
texts = fldata
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=4,random_state=100,update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)

#pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#each poem classified in either of 4 topics
lda_corpus = [max(prob[0],key=lambda y:y[1]) for prob in doc_lda]
playlists = [[] for i in range(4)]
for i, x in enumerate(lda_corpus):
    playlists[x[0]].append(dt[i])

#print in form of dataframe consisting of topic number as index and number of poems corresponding to each topic
res = pd.DataFrame(playlists)
print(res)


