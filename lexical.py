import pandas as pd
from nltk import *

################################################################
##########################DADOS#####################################

dados = pd.read_csv("base.csv", usecols = ['Textos','Arquivos'])

#print(len(dados))
#print(dados.head(5));exit(1)

Textos=dados['Textos']
#print(len(Textos))
#print(Textos[0]);exit(1)

txt=list(dados['Arquivos'])
#print(len(txt))
#print(txt);exit(1)

PS=[]
for x in txt:
    fields = x.split('@')
    fields1 = fields[1].split('_')
    fields2 = fields1[1].split('.')
    PS.append([fields2[0],int(fields1[0])])

#print(len(PS))
#print(PS);exit(1)

PconcatS=[]
for x in PS:PconcatS.append(str(x[0])+str(x[1]))

#print(len(PconcatS))
#print(PconcatS)

notas = pd.read_csv("notas.csv", usecols = ['Pacote','Sequencia','NotaF','TemaF','CoerenciaF','RegrasF'])

#print(len(notas))
#print(notas.head(5));exit(1)

notaszip = list(zip(notas['Pacote'], notas['Sequencia'],notas['NotaF'],notas['TemaF'],notas['CoerenciaF'],notas['RegrasF']))

#print(len(notaszip))
#print(notaszip[0]);exit(1)

L=[]
for x in notaszip:L.append((str(x[0])+str(x[1]),(x[2],x[3],x[4],x[5])))

#print(L[0])
#print(len(L));exit(1)

'''
dic.update({'nome': (variavel1, variavel2)})

print(dic['nome'][0])
print(dic['nome'][1]
'''
dicionario=dict(L)

print(len(dicionario))
print(dicionario);exit(1)

#[mydict[x] for x in mykeys]

NotasF=list(dicionario[x][0] for x in PconcatS)
NotasF=[int(round(4*x)) for x in NotasF]
#print(len(NotasF))
#print(NotasF)
#print(type(NotasF[0]));exit(1)

TemaF=list(dicionario[x][1] for x in PconcatS)
TemaF=[int(round(4*x)) for x in TemaF]
#print(len(TemaF))
#print(TemaF)
#print(type(TemaF[0]));exit(1)

CoerenciaF=list(dicionario[x][2] for x in PconcatS)
CoerenciaF=[int(round(4*x)) for x in CoerenciaF]
#print(len(CoerenciaF))
#print(CoerenciaF)
#print(type(CoerenciaF[0]));exit(1)

RegrasF=list(dicionario[x][3] for x in PconcatS)
RegrasF=[int(round(4*x)) for x in RegrasF]
#print(len(RegrasF))
#print(RegrasF)
#print(type(RegrasF[0]));exit(1)

#################################################################
#################################################################

from unicodedata import normalize
import textstat
from readcalc import readcalc
from lexicalrichness import *
import collections as col
class ProcLing():
    def __init__(self, lista):
        self.lista = lista

    def retiraAcentuacao(self):
        lista_sem_acentos = []
        for l in self.lista:
            try:
                l = normalize('NFKD', l).encode('ASCII','ignore').decode('ASCII')
            except:
                l = l
            lista_sem_acentos.append(l)

        return lista_sem_acentos

    def transformaMinusculas(self):
        lista_minuscula=[]
        for i in range(0,len(self.lista)):
            lista_minuscula.append(self.lista[i].lower())
        return lista_minuscula

    def retiraPontuacao(self):
        lista_sem_pontuacao=[]
        for i in range(0,len(self.lista)):
            lista_sem_pontuacao.append(re.sub(u'["...:,;()!?%&"]',' ',self.lista[i]))
        return lista_sem_pontuacao

    def filtrar(self):
        self.lista = self.retiraAcentuacao()
        self.lista = self.transformaMinusculas()
        self.lista = self.retiraPontuacao()

        return self.lista

def lenwords(T):
    L=[len(x) for x in word_tokenize(T)]
    return L

def curtas(L):
    lista_valida = []
    for elem in L:
        if elem <=4:
            lista_valida.append(elem)
    return len(lista_valida)

def longas(L):
    lista_valida = []
    for elem in L:
        if elem > 4:
            lista_valida.append(elem)
    return len(lista_valida)

def modaLista(lista):
    aux = 0
    cont = 0
    moda = -1
    lista.sort(key=None, reverse=False)
    for i in range(0, len(lista) - 1):
        if (lista[i] == lista[i + 1]):
            cont = cont + 1
            if cont >= aux:
                aux = cont
                moda = lista[i]
        else:
            cont = 0
    return moda

setSW = corpus.stopwords.words('portuguese')

def nSW(words, setSW):
    ws = []
    for w in words:
        if w in setSW: ws.append(w)
    return len(ws)

def get_yules(s):
    tokens = tokenize(s)
    token_counter = col.Counter(tok.upper() for tok in tokens)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    if m1==m2:
        i=1
    else:i = (m1*m1) / (m2-m1)
    k = 1/i * 10000
    return k

def hapaxLegonema(x):return(sum([1 for (x,y) in Counter(x).items() if y==1]))

def guiraudIndex(x):return round(len(set(x))/(len(x)**0.5)*10)

def lexical_diversity(text):return len(set(text)) / len(text)

###############################################################

nsentences=[len(sent_tokenize(x)) for x in Textos]

#print(len(nsentences))
#print(nsentences);exit(1)

filtro=ProcLing(Textos)
filtro.filtrar()
Textos=filtro.lista
#print(Textos[0]);exit(1)

ncharacteres=[]
for i in range(0,len(Textos)):ncharacteres.append(len(Textos[i]))

#print(len(ncharacteres))
#print(ncharacteres);exit(1)

nwords=[]
for i in range(0,len(Textos)):nwords.append(lenwords(Textos[i]))

#print(len(nwords))
#print(nwords[0]);exit(1)

snwords=[]
for i in range(0,len(Textos)):snwords.append(sum(nwords[i]))

#print(len(snwords))
#print(snwords);exit(1)

nlongwords=[]
for i in range(0,len(nwords)):nlongwords.append(longas(nwords[i]))

#print(len(nlongwords))
#print(nlongwords);exit(1)

nshortswords=[]
for i in range(0,len(nwords)):nshortswords.append(curtas(nwords[i]))

#print(nshortswords)
#print(len(nshortswords));exit(1)

mostfreqwordslen=[]
for i in range(0,len(nwords)):mostfreqwordslen.append(modaLista(nwords[i]))

#print(mostfreqwordslen)
#print(len(mostfreqwordslen));exit(1)

averagewordslen=[]
for i in range(0,len(nwords)):averagewordslen.append(int(sum(nwords[i])/len(nwords[i])))

#print(averagewordslen)
#print(len(averagewordslen));exit(1)

ndifwords=[]
for i in range(0,len(Textos)):ndifwords.append(len(set(Textos[i])))

#print(ndifwords)
#print(len(ndifwords));exit(1)

nstopwords=[]
for i in range(0,len(Textos)):nstopwords.append(nSW(Textos[i],setSW))

#print(len(nstopwords))
#print(nstopwords);exit(1)

nsyllable=[]
for i in range(0,len(Textos)):nsyllable.append(textstat.syllable_count(Textos[i], lang='pt_BR'))

#print(len(nsyllable))
#print(nsyllable)

SMOGindex=[]
for i in range(0,len(Textos)):
    s = readcalc.ReadCalc(Textos[i])
    SMOGindex.append(int(s.get_smog_index()))

#print(len(SMOGindex))
#print(SMOGindex);exit(1)

LD=[]
for i in range(0,len(Textos)):LD.append(lexical_diversity(Textos[i]))

LD=[int(1000*x) for x in LD]

#print(len(LD))
#print(LD);exit(1)

#type-token ratio
TTR=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    TTR.append(lex.ttr)

TTR=[int(10*x) for x in TTR]

#print(len(TTR))
#print(TTR);exit(1)

#corrected type-token ratio
CTTR=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    CTTR.append(lex.cttr)

CTTR=[int(10*x) for x in CTTR]

#print(len(CTTR))
#print(CTTR);exit(1)

#mean segmental type-token ratio

MSTTR=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    MSTTR.append(lex.msttr(segment_window=25))

MSTTR=[int(10*x) for x in MSTTR]

#print(len(MSTTR))
#print(MSTTR);exit(1)

#moving average type-token ratio

MATTR=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    MATTR.append(lex.mattr(window_size=25))

MATTR=[int(100*x) for x in MATTR]

#print(len(MATTR))
#print(MATTR);exit(1)

#Measure of Textual Lexical Diversity

MTLD=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    MTLD.append(lex.mtld(threshold=0.72))

MTLD=[int(x) for x in MTLD]

#print(len(MTLD))
#print(MTLD);exit(1)

#hypergeometric distribution diversity

HDD=[]
for i in range(0,len(Textos)):
    lex = LexicalRichness(Textos[i])
    HDD.append(lex.hdd(draws=42))

HDD=[int(100*x) for x in HDD]

#print(len(HDD))
#print(HDD);exit(1)

Gindex=[]
for i in range(0,len(Textos)):Gindex.append(guiraudIndex(Textos[i]))

#print(len(Gindex))
#print(Gindex);exit(1)

YK=[]
for i in range(0,len(Textos)):YK.append(int(get_yules(Textos[i])))

#print(len(YK))
#print(YK);exit(1)

## hapax legomena

Hapax=[]
for i in range(0,len(Textos)):Hapax.append(hapaxLegonema(Textos[i]))

#print(len(Hapax))
#print(Hapax);exit(1)

GFindex=[]
for i in range(0,len(Textos)):GFindex.append(int(textstat.gunning_fog(Textos[i])))

#print(len(GFindex))
#print(GFindex);exit(1)

FREindex=[]
for i in range(0,len(Textos)):FREindex.append(abs(int(textstat.flesch_reading_ease(Textos[i]))))

#print(len(FREindex))
#print(FREindex);exit(1)

FKGindex=[]
for i in range(0,len(Textos)):FKGindex.append(int(textstat.flesch_kincaid_grade(Textos[i])))

#print(len(FKGindex))
#print(FKGindex);exit(1)

DCindex=[]
for i in range(0,len(Textos)):DCindex.append(int(textstat.dale_chall_readability_score(Textos[i])))

#print(len(DCindex))
#print(DCindex);exit(1)

ARindex=[]
for i in range(0,len(Textos)):ARindex.append(int(textstat.automated_readability_index(Textos[i])))

#print(len(ARindex))
#print(ARindex);exit(1)

LIXindex=[]
for i in range(0,len(Textos)):
    s = readcalc.ReadCalc(Textos[i])
    LIXindex.append(int(s.get_lix_index()))

#print(len(LIXindex))
#print(LIXindex);exit(1)

CLindex=[]
for i in range(0,len(Textos)):
   s = readcalc.ReadCalc(Textos[i],language='pt',preprocesshtml=' ')
   CLindex.append(int(s.get_coleman_liau_index()))

#print(len(CLindex))
#print(CLindex);exit(1)


L=[nsentences,ncharacteres,snwords,nlongwords,nshortswords,
   mostfreqwordslen,averagewordslen,ndifwords,nstopwords,
   nsyllable,SMOGindex,LD,TTR,CTTR,MSTTR,MATTR,MTLD,
   HDD,Gindex,YK,Hapax,GFindex,FREindex,FKGindex,DCindex,
   ARindex,LIXindex,CLindex,NotasF,TemaF,CoerenciaF,RegrasF]

L=list(zip(*L))
H=[]
for i in range(len(L)):H.append(list(L[i]))

df = pd.DataFrame(H,columns=['nsentences','ncharacteres',
                             'snwords','nlongwords','nshortswords',
                             'mostfreqwordslen','averagewordslen',
                             'ndifwords','nstopwords','nsyllable',
                             'SMOGindex','LD','TTR','CTTR','MSTTR',
                             'MATTR','MTLD','HDD','Gindex','YK',
                             'Hapax','GFindex','FREindex','FKGindex',
                             'DCindex','ARindex','LIXindex','CLindex',
                             'NotasF','TemaF','CoerenciaF','RegrasF'])

df.to_csv('lexical.csv', sep=';', encoding='utf-8')


