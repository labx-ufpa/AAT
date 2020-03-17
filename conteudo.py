import pandas as pd
from nltk import *

################################################################
##########################DADOS#####################################

dados = pd.read_csv("base.csv", usecols=['Textos', 'Arquivos'])

#print(len(dados))
#print(dados.head(5));exit(1)

Textos = dados['Textos']
#print(len(Textos))
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
#print(PconcatS)

notas = pd.read_csv("notas.csv", usecols=['Pacote', 'Sequencia', 'NotaF', 'TemaF', 'CoerenciaF', 'RegrasF'])

#print(len(notas))
#print(notas.head(5));exit(1)

notaszip = list(
    zip(notas['Pacote'], notas['Sequencia'], notas['NotaF'], notas['TemaF'], notas['CoerenciaF'], notas['RegrasF']))

#print(len(notaszip))
#print(notaszip[0]);exit(1)

L = []
for x in notaszip: L.append((str(x[0]) + str(x[1]), (x[2], x[3], x[4], x[5])))

#print(L[0])
#print(len(L));exit(1)

'''
dic.update({'nome': (variavel1, variavel2)})

print(dic['nome'][0])
print(dic['nome'][1]
'''

dicionario = dict(L)

#print(len(dicionario))
#print(dicionario);exit(1)

# [mydict[x] for x in mykeys]

NotasF = list(dicionario[x][0] for x in PconcatS)
NotasF = [int(round(x)) for x in NotasF]
#print(len(NotasF))
#print(NotasF)
#print(type(NotasF[0]));exit(1)

TemaF = list(dicionario[x][1] for x in PconcatS)
TemaF = [int(round(4 * x)) for x in TemaF]
#print(len(TemaF))
#print(TemaF)
#print(type(TemaF[0]));exit(1)

CoerenciaF = list(dicionario[x][2] for x in PconcatS)
CoerenciaF = [int(round(4 * x)) for x in CoerenciaF]
#print(len(CoerenciaF))
#print(CoerenciaF)
#print(type(CoerenciaF[0]));exit(1)

RegrasF = list(dicionario[x][3] for x in PconcatS)
RegrasF = [int(round(4 * x)) for x in RegrasF]
#print(len(RegrasF))
#print(RegrasF)
#print(type(RegrasF[0]));exit(1)

arq = open('/home/joao/Documentos/jc/NovosExperimentos/Ensaios/Redacoes/source_text.txt', 'r')
source_text = arq.read()
#print(source_text);exit(1)

####################################################################
#######################PACOTES######################################
import csv
from numpy.linalg import inv

inf = float(1e-20)  # evitar divisão por zero
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

#################################################################
#################################################################

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


def cleanData(text, lowercase=False, remove_stops=False, stemming=False, lemmatization=False):
    txt = str(text)
    # txt = re.sub(r'[^A-Za-z\s]',r' ',txt)
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


def media(L): return sum(L) / len(L)


def TfIdf(matriz):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(matriz)
    return tfidf.toarray()


def tuplo(t):
    t_ord = sorted(t, reverse=True)
    L1 = t_ord[0:5]
    L2 = t_ord[5:10]
    return sum(L1) - sum(L2)


def CosV(x, y): return round(abs(np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)), 2)


def euclD(x, y): return round(np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y))), 2)


def LSA(d, y, v):
    L = []
    for i in range(0, len(d)):
        vocabulary = [v, d[i]]
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
        dtm = vectorizer.fit_transform(vocabulary)
        df = pd.DataFrame(dtm.toarray(), index=vocabulary, columns=vectorizer.get_feature_names()).head(len(d))
        m = df.values.T.tolist()
        matriz = np.matrix(m)
        matriz=TfIdf(matriz)
        U, Sigma, Vt = linalg.svd(matriz)
        k = 2
        S = np.array(Sigma[0:k])
        Sk = np.diag(S)
        Vtk = Vt[:, 0:k]
        Vtkt = Vtk.transpose()
        Ak = np.dot(Sk, Vtkt)
        L1 = []
        if y == str('cosseno'):
            L1.append(abs(CosV(Ak[:, 0], Ak[:, 1])))
        elif y == str('distancia'):
            L1.append(euclD(Ak[:, 0], Ak[:, 1]))
        L.append(L1[0])
    return L

def RLM(a, b, c):
    u = np.ones(len(a))
    L = [u, a, b]
    M = np.array(L)
    N = M.transpose()
    N1 = inv(N.transpose().dot(N))
    N2 = N.transpose().dot(c)
    b = N1.dot(N2)
    return b

import statsmodels.api as sm

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
        results = sm.OLS(y, X).fit()
    return results

def inter(L,y1,y2):
    l=[]
    for x in L:l.append(y1+(x-min(L))/((y2-y1)/(max(L)-min(L))))
    return l
#################################################################
################################################################

Textos = retiraAcentuacao(Textos)
Textos = transformaMinusculas(Textos)
Textos = retiraPontuacao(Textos)
#print(Textos[0])
#print(len(Textos));exit()

'''
Textos_ssw=[]
for i in range(0,len(Textos)):
    Textos_ssw.append(cleanData(Textos[i],remove_stops = True))

Textos=Textos_ssw

Textos_cst=[]
for i in range(0,len(Textos)):
    Textos_cst.append(cleanData(Textos_ssw[i],stemming = True))

Textos=Textos_cst
'''

###############################################################
###############################################################

TN = list(zip(Textos, NotasF))
#print(TN[0])

random.seed(42);
random.shuffle(TN)
#print(TN[0])
#exit(1)

TextosS, NotasS = zip(*TN)
TextosS = list(TextosS);
#print(len(TextosS))
#print(TextosS[0])
NotasS = list(NotasS)
#print(len(NotasS))
#print(NotasS[0]);exit(1)

index1=random.randint(0,len(TextosS)-1)
del (TextosS[index1])
del (NotasS[index1])
#print(len(TextosS))
#print(len(NotasS));exit(1)

index2=random.randint(0,len(TextosS)-1)
del (TextosS[index2])
del (NotasS[index2])
#print(len(TextosS))
#print(len(NotasS));exit(1)

#####################################################################
#####################################################################

def modelo():
    #print('modelo...')
    mod1f = " ".join(textos_treina[i] for i in range(len(textos_treina)) if notas_treina[i] <= 3.0)
    #print(mod1f);exit(1)

    mod2f = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 3.0 <= notas_treina[i] <= 6.0)
    #print(mod2f);exit(1)

    mod3f = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 5.0 <= notas_treina[i] <= 8.0)
    #print(mod3f);exit(1)

    mod4f = " ".join(textos_treina[i] for i in range(len(textos_treina)) if notas_treina[i] >=7.0)
    #print(mod4f);exit(1)

    #mod1 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if notas_treina[i] <= 2.5)
    #print(mod1);exit(1)

    #mod2 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 1.0 <= notas_treina[i] <= 2.5)
    #print(mod2);exit(1)

    #mod3 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 1.0 < notas_treina[i] <= 3.0)
    #print(mod3);exit(1)

    #mod4 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 2.0 <= notas_treina[i] <= 4.0)
    #print(mod4);exit(1)

    #mod5 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 3.0 <= notas_treina[i] <= 5.0)
    #print(mod5);exit(1)

    #mod6 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 4.0 <= notas_treina[i] <= 6.0)
    #print(mod6);exit(1)

    #mod7 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 5.0 <= notas_treina[i] <= 7.0)
    #print(mod7);exit(1)

    #mod8 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 6.0 <= notas_treina[i] <= 8.0)
    #print(mod8);exit(1)

    #mod9 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 7.0 <= notas_treina[i] <= 9.0)
    #print(mod9);exit(1)

    #mod10 = " ".join(textos_treina[i] for i in range(len(textos_treina)) if 8.0 <= notas_treina[i] <= 10.0)
    #print(mod10);exit(1)

    modT = " ".join(textos_treina[i] for i in range(len(textos_treina)))
    #print(modT);exit(1)

    modTT = " ".join(TextosS[i] for i in range(len(TextosS)))
    #print(modTT);exit(1)

    modes = [mod1f, mod2f, mod3f, mod4f,modT, modTT]

    global Hum;
    Hum = [int(x) for x in notas_teste]
    #print(len(Hum))
    #print(max(Hum), min(Hum));exit(1)

    global simc;
    simc = []
    for m in modes:
        sim_1 = LSA(textos_teste, 'cosseno', m)
        #print(sim_1);exit(1)
        simc.append([int(100*x) for x in sim_1])
    print(len(simc[0]))
    print(simc[0])
    H=inter(simc[0],min(Hum),max(Hum))
    print(H);exit(1)
    #print(len(simc))
    #print(simc);exit(1)

    global sim_sourcetextc;
    sim_sourcetextc = LSA(textos_teste, 'cosseno', source_text)
    sim_sourcetextc = [int(100 * x) for x in sim_sourcetextc]
    #print(len(sim_sourcetextc))
    #print(sim_sourcetextc);exit(1)

    global simd;
    simd = []
    for m in modes:
        sim_1 = LSA(textos_teste, 'distancia', m)
        simd.append([int(round(4*x/10)) for x in sim_1])
    #print(len(simd[0]))
    #print(simd[0])
    #print(len(simd))
    #print(simd);exit(1)


    global sim_sourcetextd;
    sim_sourcetextd = LSA(textos_teste, 'distancia', source_text)
    sim_sourcetextd = [int(x) for x in sim_sourcetextd]
    #print(len(sim_sourcetextd))
    #print(sim_sourcetextd);exit(1)

    x0=[simc[0],simd[0]]
    R0=reg_m(Hum, x0).params
    RR0=[simc[0][i] * R0[0] + simd[0][i] * R0[1] + R0[2] for i in range(len(simc[0]))]
    RR0=[int(round(x)) for x in RR0]
    #print(R0)
    #print(len(RR0))
    #print(RR0);exit(1)

    x1 = [simc[1], simd[1]]
    R1 = reg_m(Hum, x1).params
    RR1 = [simc[1][i] * R1[0] + simd[1][i] * R1[1] + R1[2] for i in range(len(simc[1]))]
    RR1 = [int(round(x)) for x in RR1]
    #print(R1)
    #print(len(RR1))
    #print(RR1);exit(1)

    x2 = [simc[2], simd[2]]
    R2 = reg_m(Hum, x2).params
    RR2 = [simc[2][i] * R2[0] + simd[2][i] * R2[1] + R2[2] for i in range(len(simc[2]))]
    RR2 = [int(round(x)) for x in RR2]
    #print(R2)
    #print(len(RR2))
    #print(RR2);exit(1)

    x3 = [simc[3], simd[3]]
    R3 = reg_m(Hum, x3).params
    RR3 = [simc[3][i] * R3[0] + simd[3][i] * R3[1] + R3[2] for i in range(len(simc[3]))]
    RR3 = [int(round(2*x/5)) for x in RR3]
    #print(R3)
    #print(len(RR3))
    #print(RR3);exit(1)

    x4 = [simc[4], simd[4]]
    R4 = reg_m(Hum, x4).params
    RR4 = [simc[4][i] * R4[0] + simd[4][i] * R4[1] + R4[2] for i in range(len(simc[4]))]
    RR4 = [int(round(2*x/5)) for x in RR4]
    #print(R4)
    #print(len(RR4))
    #print(RR4);exit(1)

    x5 = [simc[5], simd[5]]
    R5 = reg_m(Hum, x5).params
    RR5 = [simc[5][i] * R5[0] + simd[5][i] * R5[1] + R5[2]/10 for i in range(len(simc[5]))]
    RR5 = [int(round(2 * x / 5)) for x in RR5]
    #print(R5)
    #print(len(RR5))
    #print(RR5);exit(1)

    global simcd;
    simcd = [RR0,RR1,RR2,RR3,RR4,RR5]
    #print(len(simcd))
    #print(simcd);exit(1)

    global sim_sourcetextcd;
    y=[sim_sourcetextc,sim_sourcetextd]
    RST = reg_m(Hum, y).params
    #print(RST);exit(1)
    sim_sourcetextcd = [sim_sourcetextc[i]*RST[0]+sim_sourcetextd[i]*RST[1]+RST[0]/5for i in range(len(sim_sourcetextc))]
    sim_sourcetextcd = [int(round((x)))  for x in sim_sourcetextcd]
    #print(len(sim_sourcetextcd))
    #print(sim_sourcetextcd);exit(1)

def grava():
    sim = []
    sim.extend(simc)
    sim.append(sim_sourcetextc)
    sim.append(Hum)
    L = sim
    L = list(zip(*L))
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    cols = 'f1c,f2c,f3c,f4c,ct,ctt,cst,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'content' + str(faixa) + 'c.csv'
    df.to_csv(filex)

    sim = []
    sim.extend(simd)
    sim.append(sim_sourcetextd)
    sim.append(Hum)
    L = sim
    L = list(zip(*L))
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    cols = 'f1d,f2d,f3d,f4d,dt,dtt,dst,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'content' + str(faixa) + 'd.csv'
    df.to_csv(filex)

    sim = []
    sim.extend(simcd)
    sim.append(sim_sourcetextcd)
    sim.append(Hum)
    L = sim
    L = list(zip(*L))
    H = []
    for i in range(len(L)): H.append(list(L[i]))
    cols = 'f1cd,f2cd,f3cd,f4cd,cdt,cdtt,cdst,Hum'.split(',')
    df = pd.DataFrame(H, columns=[*cols])
    filex = 'content' + str(faixa) + 'cd.csv'
    df.to_csv(filex)

textos_teste = []
textos_treina = []
notas_teste = []
notas_treina = []
passo = 50
for faixa in range(0, 1000, passo):
    print(faixa, faixa + passo)
    textos_teste = TextosS[faixa:faixa + passo]
    #print(textos_teste);exit(1)
    notas_teste = NotasS[faixa:faixa + passo]
    textos_treina = TextosS[0:faixa] + TextosS[faixa + passo:]
    notas_treina = NotasS[0:faixa] + NotasS[faixa + passo:]
    #print('teste:', textos_teste[0][:39])
    #print(len(textos_teste))
    #print('treina:', textos_treina[0][:39])
    #print(len(textos_treina))
    modelo()
    grava()
exit(1)

