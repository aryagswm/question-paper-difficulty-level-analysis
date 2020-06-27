import nltk
import yaml
from check import df
import pandas as pd
class Splitter(object):
    def __init__(self):
        self.nltk_splitter=nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer=nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        sentences=self.nltk_splitter.tokenize(text)
        tokenized_sentences=[self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):
    def __init__(self):
        pass

    def pos_tag(self,sentences):
        pos=[nltk.pos_tag(sentence) for sentence in sentences]
        #print("pos in function",pos)
        pos=[[(word,word,[postag]) for (word,postag) in sentence] for sentence in pos]
        return pos



class DictionaryTagger(object):
    def __init__(self,dictionary_paths):
        files=[open(path,'r') for path in dictionary_paths]
        dictionaries=[yaml.load(dict_file) for dict_file in files]
        map(lambda x:x.close(),files)
        self.dictionary={}
        self.max_key_size=0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key]=curr_dict[key]
                    self.max_key_size=max(self.max_key_size,len(key))

    def tag(self,postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self,sentence,tag_with_lemmas=False):
        tag_sentence=[]
        n=len(sentence)
        if self.max_key_size==0:
            self.max_key_size=n

        i=0

        while(i<n):
            j=min(i+self.max_key_size,n)
            tagged=False
            while(j>i):
                expression_form=' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma=' '.join([word[1] for word in sentence[i:j]]).lower()
                #print(sentence[i:j])
                #print("expression form and lemma")
                #print(expression_form)
                #print(expression_lemma)
                if tag_with_lemmas:
                    literal=expression_lemma
                else:
                    literal=expression_form
                if literal in self.dictionary:
                    is_single_token=j-i==1
                    original_position=i
                    i=j
                    taggings=[tag for tag in self.dictionary[literal]]
                    tagged_expression=(expression_form,expression_lemma,taggings)
                    if is_single_token:
                        original_token_tagging=sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged=True
                else:
                    j=j-1
            if not tagged:
                tag_sentence.append(sentence[i])
                i+=1
        return tag_sentence



def value_of(sentiment):
    if sentiment=='positive':return 1
    if sentiment == 'negative': return -1
    return 0
previous_tags=[]
def sentence_score(sentence_tokens, previous_token, acum_score):
    #print("sentence tokens",sentence_tokens)
    #print("acum_score",acum_score)
    if not sentence_tokens:
        return acum_score
    else:
        current_token = sentence_tokens[-1]
        tags = current_token[2]
        token_score = float(sum([value_of(tag) for tag in tags]))
        acum_score=token_score+acum_score
        #print("original",token_score)
        if previous_token is not None:
            previous_tags =previous_token[2]
            '''if previous_token[2] not in previous_tags:
                previous_tags.append(previous_token[2])
            for tag in previous_tags:
                #print("previous_tags",previous_tags)'''
            if 'inc' in previous_tags:
                acum_score *= 2.0
                #print("inc",token_score)
            if 'dec' in previous_tags:
                acum_score /= 2.0
                #print("dec",token_score)
            if 'inv' in previous_tags:
                acum_score*=-1
                #print("inv",token_score)
        #print("final",token_score)
        return sentence_score(sentence_tokens[:-1], current_token, acum_score)

def sentiment_score(review):
    return sum([sentence_score(sentence,None,0.0) for sentence in review])

splitter=Splitter()
pos_tagger=POSTagger()
dicttagger = DictionaryTagger([ 'positive.yml', 'negative.yml','inc.yml','dec.yml','inv.yml'])
#text=input("enter text")
complete=pd.DataFrame({'Score':[],'Remarks':[]})
for text in df.Remarks:
    #print("here")
    splitted_sentence=splitter.split(text)
    #print(splitted_sentence)

    pos_tagged_sentences=pos_tagger.pos_tag(splitted_sentence)
    #print(pos_tagged_sentences)


    dict_tagged_sentences=dicttagger.tag(pos_tagged_sentences)
    #print(dict_tagged_sentences)



    #print(sentiment_score(dict_tagged_sentences))
    single=pd.DataFrame({'Score':[sentiment_score(dict_tagged_sentences)],'Remarks':[text]})
    complete=pd.concat([complete,single])


complete.to_csv("score.csv",index=False)


