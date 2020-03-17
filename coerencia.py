####################################################################
#######################PACOTES######################################
import csv
from numpy.linalg import inv
inf=float(1e-20) # evitar divisão por zero
import re
from unicodedata import normalize
import nltk
from nltk import *

from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
import random
#####################################################################
########################FUNCOES#####################################
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
            lista_sem_pontuacao.append(re.sub(u'["-...:,;()!?%&"]',' ',self.lista[i]))
        return lista_sem_pontuacao

    def filtrar(self):
        self.lista = self.retiraAcentuacao()
        self.lista = self.transformaMinusculas()
        self.lista = self.retiraPontuacao()

        return self.lista

stop_words = set(stopwords.words('portuguese'))

def retiraAcentuacao(L):
    lista_sem_acentos = []
    for l in L:
        try:
            l = normalize('NFKD', l).encode('ASCII', 'ignore').decode('ASCII')
        except:
            l = l
        lista_sem_acentos.append(l)
    return lista_sem_acentos

def transformaMinusculas(L):
    lista_minuscula = []
    for i in range(0, len(L)):
        lista_minuscula.append(L[i].lower())
    return lista_minuscula

def retiraPontuacao(L):
    lista_sem_pontuacao = []
    for i in range(0, len(L)):
        lista_sem_pontuacao.append(re.sub(u'["-...:,;()!?%&"]', ' ', L[i]))
    return lista_sem_pontuacao

def cleanData(text, lowercase = False, remove_stops = False, stemming = False, lemmatization = False):
    txt = str(text)
    #txt = re.sub(r'[^A-Za-z\s]',r' ',txt)
    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])
    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stop_words])
    if stemming:
        st = RSLPStemmer()
        txt = " ".join([st.stem(w) for w in txt.split()])
    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])
    return txt

def media(L): return sum(L)/len(L)

def TfIdf(matriz):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(matriz)
    return tfidf.toarray()

def tuplo(t):
    t_ord=sorted(t,reverse=True)
    L1=t_ord[0:5]
    L2=t_ord[5:10]
    return sum(L1)-sum(L2)

def CosV(x,y): return round(abs(np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y)),2)

def euclD(x,y):return round(np.sqrt(sum(pow(a-b,2) for a, b in zip(x, y))),2)

def LSA(d,y,v):
    L=[]
    for i in range(0,len(d)):
        vocabulary=[v,d[i]]
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
        dtm = vectorizer.fit_transform(vocabulary)
        df = pd.DataFrame(dtm.toarray(), index=vocabulary, columns=vectorizer.get_feature_names()).head(len(d))
        m = df.values.T.tolist()
        matriz = np.matrix(m)
        #matriz=TfIdf(matriz)
        U, Sigma, Vt = linalg.svd(matriz)
        k=2
        S = np.array(Sigma[0:k])
        Sk = np.diag(S)
        Vtk = Vt[:, 0:k]
        Vtkt = Vtk.transpose()
        Ak = np.dot(Sk, Vtkt)
        L1=[]
        if y==str('cosseno'):L1.append(abs(CosV(Ak[:, 0],Ak[:, 1])))
        elif y==str('distancia'):L1.append(euclD(Ak[:, 0],Ak[:, 1]))
        L.append(L1[0])
    return L

def RLM(a,b,c):
    u=np.ones(len(a))
    L=[u,a,b]
    M = np.array(L)
    N = M.transpose()
    N1 = inv(N.transpose().dot(N))
    N2 = N.transpose().dot(c)
    b = N1.dot(N2)
    return b

def wind(base,lenght,txt):
    x1,x2=base,base+lenght;
    w1=txt[x1:x2];
    return w1

## coleta todas as janelas
## a ultima incompleta eh desprezada
maxi = 5
def getWind(txt):
    b=0;W=[];
    lenght=round(len(txt)/4)
    while b+lenght<=len(txt):
        w1=wind(b,lenght,txt);W.append(w1);b+=maxi
    if len(W)==1: W+=[txt,['0','0']]
    #w1,w2=wind(b,t1);W.append(w1)
    return W

def windows(txt):
    X1 = word_tokenize(txt)
    L = getWind(X1)
    M = []
    for j in range(0, len(L)):
        M.append(" ".join(L[j]))
    return M

def limpa(x):return re.sub(u'["-,.!?%&"]',' ',x)

def LSACONTIGUOS(d,y):
    L=[]
    for i in range(0,len(d)-1):
        vocabulary=[d[i],d[i+1]]
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
        dtm = vectorizer.fit_transform(vocabulary)
        df = pd.DataFrame(dtm.toarray(), index=vocabulary, columns=vectorizer.get_feature_names()).head(len(d))
        m = df.values.T.tolist()
        matriz = np.matrix(m)
        matriz=TfIdf(matriz)
        U, Sigma, Vt = linalg.svd(matriz)
        k=int(np.sqrt(len(d)))
        S = np.array(Sigma[0:k])
        Sk = np.diag(S)
        Vtk = Vt[:, 0:k]
        Vtkt = Vtk.transpose()
        Ak = np.dot(Sk, Vtkt)
        L1=[]
        if y==str('cosseno'):L1.append(abs(CosV(Ak[:, 0],Ak[:, 1])))
        elif y==str('distancia'):L1.append(euclD(Ak[:, 0],Ak[:, 1]))
        L.append(L1[0])
    return L

####################################################################
#######################DADOS######################################
dados = pd.read_csv("base.csv", usecols=['Textos', 'Arquivos'])

#print(len(dados))
#print(dados.head(5));exit(1)

T = dados['Textos']
#print(len(T))

Textos = []
for i in range(0, len(T)): Textos.append(T[i])
#print(Textos[0]);exit(1)

txt = list(dados['Arquivos'])
#print(len(txt))
#print(txt);exit(1)

PS = []
for x in txt:
    fields = x.split('@')
    fields1 = fields[1].split('_')
    fields2 = fields1[1].split('.')
    PS.append([fields2[0], int(fields1[0])])

#print(len(PS))
#print(PS);exit(1)

PconcatS = []
for x in PS: PconcatS.append(str(x[0]) + str(x[1]))

#print(len(PconcatS))
#print(PconcatS);exit(1)

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

dicionario = dict(L)

#print(len(dicionario))
#print(dicionario);exit(1)

# [mydict[x] for x in mykeys]

Notas = list(dicionario[x][0] for x in PconcatS)

TemaF=list(dicionario[x][1] for x in PconcatS)

CoerenciaF=list(dicionario[x][2] for x in PconcatS)

RegrasF=list(dicionario[x][3] for x in PconcatS)

#exit(1)

####################################################################
#######################TESTE X TREINAMENTO######################################

Textos=retiraAcentuacao(Textos)
Textos=transformaMinusculas(Textos)
Textos=retiraPontuacao(Textos)
#print(Textos[0])
#print(len(Textos));exit()

filtro=ProcLing(Textos)
Textos=filtro.lista

#print(Textos[0])
#print(len(Textos));exit()

TN=list(zip(Textos,RegrasF))
#print(TN[0]);exit(1)

#random.seed(42);random.shuffle(TN)
#print(TN[0]);exit(1)

TextosS,NotasS =zip(*TN)
TextosS=list(TextosS);NotasS=list(NotasS)
#print(len(TextosS))
#print(max(NotasS));exit(1)

del(TextosS[0])
del(NotasS[0])
del(TextosS[1])
del(NotasS[1])

#print(len(TextosS))
#print(len(NotasS));exit(1)

textos_teste = []
textos_treina = []
notas_teste = []
notas_treina= []

# contíguos

'''
def modelo():
    print('modelo...')
    TextosWind = []
    for i in range(len(textos_teste)): TextosWind.append(windows(textos_teste[i]))
    CC1c = []
    for i in range(0, len(TextosWind)): CC1c.append(LSACONTIGUOS(TextosWind[i], 'cosseno'))
    modes=[max,min,media]
    global simc;simc = []
    for i in range(0,len(CC1c)):
        L=[]
        for m in modes:
           L.append(m(CC1c[i]))
        simc.append(L)
    #print(len(sim_contiguosc))
    #print(sim_contiguosc);exit(1)
    res = list(zip(*simc))
    #print(len(res))
    #print(res);exit(1)
    global sim_contiguosc;sim_contiguosc=[]
    for i in range(len(res)):
        sim_contiguosc.append([int(v*100) for v in res[i]])
    #print(sim_contiguosc);exit(1)
    CC1d = []
    for i in range(0, len(TextosWind)): CC1d.append(LSACONTIGUOS(TextosWind[i], 'distancia'))
    modes = [max, min, media]

    global simd;simd = []
    for i in range(0, len(CC1d)):
        L = []
        for m in modes:
            L.append(m(CC1d[i]))
        simd.append(L)
    res = list(zip(*simd))
    global sim_contiguosd;sim_contiguosd = []
    for i in range(len(res)):
        sim_contiguosd.append([int(v) for v in res[i]])
    #print(len(sim_contiguosd[0]));exit(1)

    global Hum;Hum = [int(4 * x) for x in notas_teste]
    #print(len(Hum));exit(1)
    b = []
    #for i in range(0, len(sim_contiguosc)):
    #    l = RLM(sim_contiguosc[i], sim_contiguosd[i], Hum)
        #print(l);exit(1)
    #    b.append(l)

    #global sim_contiguoscd; sim_contiguoscd = []
    #for i in range(0, len(sim_contiguosd)):
    #    L = []
    #    for j in range(0, len(sim_contiguosd[i])):
    #        k = b[i][0] + b[i][1] * sim_contiguosc[i][j] + b[i][2] * sim_contiguosd[i][j]
    #        L.append(int(round(k)))
    #    sim_contiguoscd.append(L)
    #print(len(sim_contiguoscd));exit(1)

def grava():
    sim=[]
    sim.extend(sim_contiguosc)
    sim.append(Hum)
    L=sim
    L=list(zip(*L))
    H=[]
    for i in range(len(L)):H.append(list(L[i]))
    cols='cgc1,cgc2,cgc3,Notas'.split(',')
    df = pd.DataFrame(H,columns=[*cols])
    filex='contiguos'+str(faixa)+'c.csv'
    df.to_csv(filex)

    sim=[]
    sim.extend(sim_contiguosd)
    sim.append(Hum)
    L=sim
    L=list(zip(*L))
    H=[]
    for i in range(len(L)):H.append(list(L[i]))
    cols='cgd1,cgd2,cgd3,Notas'.split(',')
    df = pd.DataFrame(H,columns=[*cols])
    filex='contiguos'+str(faixa)+'d.csv'
    df.to_csv(filex)
    
    #sim=[]
    #sim.extend(sim_contiguoscd)
    #sim.append(Hum)
    #L=sim
    #L=list(zip(*L))
    #H=[]
    #for i in range(len(L)):H.append(list(L[i]))
    #cols='cgcd1,cgcd2,cgcd3,Notas'.split(',')
    #df = pd.DataFrame(H,columns=[*cols])
    #filex='contiguos'+str(faixa)+'cd.csv'
    #df.to_csv(filex)
    
passo=50
for faixa in range(0,1000,passo):
    print(faixa, faixa+passo)
    #exit(1)
    textos_teste= TextosS[faixa:faixa+passo]
    notas_teste= NotasS[faixa:faixa+passo]
    textos_treina= TextosS[0:faixa]+TextosS[faixa+passo:]
    notas_treina= NotasS[0:faixa]+NotasS[faixa+passo:]
    print('teste:', textos_teste[0][:39])
    print(len(textos_teste))
    print('treina:', textos_treina[0][:39])
    print(len(textos_treina))
    modelo()
    grava()
exit(1)
'''
#Todos x todos

'''
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
lsa = TruncatedSVD(2, algorithm='randomized')

def modelo():
    print('modelo...')
    TextosWind = []
    for i in range(len(textos_teste)): TextosWind.append(windows(textos_teste[i]))
    TT1 = []
    for i in range(0, len(TextosWind)):
        vocabulary = TextosWind[i]
        dtm = vectorizer.fit_transform(vocabulary)
        dtm_lsa = lsa.fit_transform(dtm)
        dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
        similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
        df = pd.DataFrame(similarity, index=vocabulary, columns=vocabulary).head(len(TextosWind[i]))
        m, n = df.shape
        df = df.values
        TT1.append(list(df[np.triu_indices(m, k=1)]))

    modes = [max, min, media]
    global sim;sim = []
    for i in range(0, len(TT1)):
        L = []
        for m in modes:
            L.append(m(TT1[i]))
        sim.append(L)

    res = list(zip(*sim))
    # print(len(res))
    # print(res);exit(1)
    global sim_all;sim_all = []
    for i in range(len(res)):
        sim_all.append([int(v * 100) for v in res[i]])
    #print(sim_all);exit(1)
    global Hum;Hum = [int(4 * x) for x in notas_teste]

def grava():
    sim=[]
    sim.extend(sim_all)
    sim.append(Hum)
    L=sim
    L=list(zip(*L))
    H=[]
    for i in range(len(L)):H.append(list(L[i]))
    cols='tdc1,tdc2,tdc3,Hum'.split(',')
    df = pd.DataFrame(H,columns=[*cols])
    filex='all'+str(faixa)+'.csv'
    df.to_csv(filex)

passo=50
for faixa in range(0,1000,passo):
    print(faixa, faixa+passo)
    #exit(1)
    textos_teste= TextosS[faixa:faixa+passo]
    notas_teste= NotasS[faixa:faixa+passo]
    textos_treina= TextosS[0:faixa]+TextosS[faixa+passo:]
    notas_treina= NotasS[0:faixa]+NotasS[faixa+passo:]
    print('teste:', textos_teste[0][:39])
    print(len(textos_teste))
    print('treina:', textos_treina[0][:39])
    print(len(textos_treina))
    modelo()
    grava()
exit(1)
'''


# centro local

from collections import Counter as fdCount
from nltk.corpus import stopwords
stop_words = set(stopwords.words('portuguese'))
def clearSW(TXT): return [w for w in TXT if not w in stop_words]

FreqDist = fdCount
def mkCentroids(txt): ## CENTROIDS
    TXT = ''.join(txt);
    TXT = word_tokenize(TXT.lower())
    TXT_= clearSW(TXT)
    FFF= list(FreqDist(TXT_).items())
    FFF.sort(key=lambda x: x[1])
    CENTR2=  FFF[-1000:]
    CENTR2,_=zip(* CENTR2)
    return list(CENTR2)

def modelo():
    print('modelo...')
    TextosWind = []
    for i in range(len(textos_teste)): TextosWind.append(windows(textos_teste[i]))
    CL = []
    for i in range(0, len(TextosWind)):
        centro_local = ' '.join(mkCentroids(TextosWind[i]))
        CL.append(centro_local)
    #print(len(CL[0]));exit(1)
    simca=[]
    for i in range(0,len(CL)):
        sim_1 = LSA(textos_teste, 'cosseno', CL[i])
        simca.append([int(100 * x) for x in sim_1])
    #print(len(simca))
    #print(simca[0]);exit(1)

    modes = [max, min, media]
    global simc;simc = []
    for i in range(0, len(simca)):
        L = []
        for m in modes:
            L.append(m(simca[i]))
        simc.append(L)
    #print(len(simc))
    #print(simc[0]);exit(1)

    global simda;simda = []
    for i in range(0, len(CL)):
        sim_1 = LSA(textos_teste, 'distancia', CL[i])
        simda.append([int(x) for x in sim_1])
    #print(len(simda))
    #print(simda[0]);exit(1)

    global simd;simd = []
    for i in range(0, len(simda)):
        L = []
        for m in modes:
            L.append(m(simda[i]))
        simd.append(L)

    #print(len(simd))
    #print(simd[0]);exit(1)

    global Hum;
    Hum = [int(4 * x) for x in notas_teste]
    #print(len(Hum))
    #print(Hum);exit(1)

    b = []
    for i in range(0, len(simc)):
        l = RLM(simca[i], simda[i], Hum)
        b.append(l)

    simcda = []
    for i in range(0, len(simc)):
        L = []
        for j in range(0, len(simca[i])):
            k = b[i][0] + b[i][1] * simca[i][j] + b[i][2] * simda[i][j]
            L.append(int(round(k)))
        simcda.append(L)

    #print(len(simcda))
    #print(simcda[0]);exit(1)

    global simcd;simcd = []
    for i in range(0, len(simcda)):
        L = []
        for m in modes:
            L.append(m(simcda[i]))
        simcd.append(L)

    #print(len(simcd))
    #print(simcd);exit(1)

def grava():
    sim=list(zip(*simc))
    #print(sim[0])
    #print(len(sim));exit(1)
    sim.append(Hum)
    L=sim
    #print(len(L));exit(1)
    #print(type(L1[0]));exit(1)
    H=[]
    for i in range(len(L)):H.append(list(L[i]))
    #print('H:',H[0])
    #print(len(H));
    H=list(zip(*H))
    cols='localcenterc1,localcenterc2,localcenterc3,Hum'.split(',')
    df = pd.DataFrame(H,columns=[*cols])
    filex='local'+str(faixa)+'c.csv'
    df.to_csv(filex)

    sim = list(zip(*simd))
    sim.append(Hum)
    L = sim
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    H = list(zip(*H))
    cols = 'localcenterd1,localcenterd2,localcenterd3,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'local' + str(faixa) + 'd.csv'
    df.to_csv(filex)

    sim = list(zip(*simcd))
    sim.append(Hum)
    L = sim
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    H = list(zip(*H))
    cols = 'localcentercd1,localcentercd2,localcentercd3,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'local' + str(faixa) + 'cd.csv'
    df.to_csv(filex)

passo=50
for faixa in range(0,1000,passo):
    print(faixa, faixa+passo)
    #exit(1)
    textos_teste= TextosS[faixa:faixa+passo]
    notas_teste= NotasS[faixa:faixa+passo]
    textos_treina= TextosS[0:faixa]+TextosS[faixa+passo:]
    notas_treina= NotasS[0:faixa]+NotasS[faixa+passo:]
    print('teste:', textos_teste[0][:39])
    print(len(textos_teste))
    print('treina:', textos_treina[0][:39])
    print(len(textos_treina))
    modelo()
    grava()
exit(1)


'''
# centro global

from collections import Counter as fdCount
from nltk.corpus import stopwords
stop_words = set(stopwords.words('portuguese'))
def clearSW(TXT): return [w for w in TXT if not w in stop_words]

FreqDist = fdCount
def mkCentroids(txt): ## CENTROIDS
    TXT = ''.join(txt);
    TXT = word_tokenize(TXT.lower())
    TXT_= clearSW(TXT)
    FFF= list(FreqDist(TXT_).items())
    FFF.sort(key=lambda x: x[1])
    CENTR2=  FFF[-1000:]
    CENTR2,_=zip(* CENTR2)
    return list(CENTR2)

def modelo():
    print('modelo...')
    TextosWind = []
    for i in range(len(textos_teste)): TextosWind.append(windows(textos_teste[i]))
    CL = []
    for i in range(0, len(TextosWind)):
        centro_local = ' '.join(mkCentroids(TextosWind[i]))
        CL.append(centro_local)
    #print(len(CL[0]));exit(1)
    simca=[]
    for i in range(0,len(CL)):
        sim_1 = LSA(textos_teste, 'cosseno', CL[i])
        simca.append([int(100 * x) for x in sim_1])
    #print(len(simca))
    #print(simca[0]);exit(1)

    modes = [max, min, media]
    global simc;simc = []
    for i in range(0, len(simca)):
        L = []
        for m in modes:
            L.append(m(simca[i]))
        simc.append(L)
    #print(len(simc))
    #print(simc[0]);exit(1)

    global simda;simda = []
    for i in range(0, len(CL)):
        sim_1 = LSA(textos_teste, 'distancia', CL[i])
        simda.append([int(x) for x in sim_1])
    #print(len(simda))
    #print(simda[0]);exit(1)

    global simd;simd = []
    for i in range(0, len(simda)):
        L = []
        for m in modes:
            L.append(m(simda[i]))
        simd.append(L)

    #print(len(simd))
    #print(simd[0]);exit(1)

    global Hum;
    Hum = [int(4 * x) for x in notas_teste]
    #print(len(Hum))
    #print(Hum);exit(1)

    b = []
    for i in range(0, len(simc)):
        l = RLM(simca[i], simda[i], Hum)
        b.append(l)

    simcda = []
    for i in range(0, len(simc)):
        L = []
        for j in range(0, len(simca[i])):
            k = b[i][0] + b[i][1] * simca[i][j] + b[i][2] * simda[i][j]
            L.append(int(round(k)))
        simcda.append(L)

    #print(len(simcda))
    #print(simcda[0]);exit(1)

    global simcd;simcd = []
    for i in range(0, len(simcda)):
        L = []
        for m in modes:
            L.append(m(simcda[i]))
        simcd.append(L)

    #print(len(simcd))
    #print(simcd);exit(1)

def grava():
    sim=list(zip(*simc))
    #print(sim[0])
    #print(len(sim));exit(1)
    sim.append(Hum)
    L=sim
    #print(len(L));exit(1)
    #print(type(L1[0]));exit(1)
    H=[]
    for i in range(len(L)):H.append(list(L[i]))
    #print('H:',H[0])
    #print(len(H));
    H=list(zip(*H))
    cols='localcenterc1,localcenterc2,localcenterc3,Hum'.split(',')
    df = pd.DataFrame(H,columns=[*cols])
    filex='local'+str(faixa)+'c.csv'
    df.to_csv(filex)

    sim = list(zip(*simd))
    sim.append(Hum)
    L = sim
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    H = list(zip(*H))
    cols = 'localcenterd1,localcenterd2,localcenterd3,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'local' + str(faixa) + 'd.csv'
    df.to_csv(filex)

    sim = list(zip(*simcd))
    sim.append(Hum)
    L = sim
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    H = list(zip(*H))
    cols = 'localcentercd1,localcentercd2,localcentercd3,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'local' + str(faixa) + 'cd.csv'
    df.to_csv(filex)

passo=50
for faixa in range(0,1000,passo):
    print(faixa, faixa+passo)
    #exit(1)
    textos_teste= TextosS[faixa:faixa+passo]
    notas_teste= NotasS[faixa:faixa+passo]
    textos_treina= TextosS[0:faixa]+TextosS[faixa+passo:]
    notas_treina= NotasS[0:faixa]+NotasS[faixa+passo:]
    print('teste:', textos_teste[0][:39])
    print(len(textos_teste))
    print('treina:', textos_treina[0][:39])
    print(len(textos_treina))
    modelo()
    grava()
exit(1)
'''