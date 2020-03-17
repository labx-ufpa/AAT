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

dicionario=dict(L)

#print(len(dicionario))
#print(dicionario);exit(1)

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

from Aelius import AnotaCorpus as a
from Aelius.Toqueniza import TOK_PORT as tok
import pandas as pd
from nltk import *

def extrai_tokens(pares):
    return [w for w,t in pares]

def extrai_tags(pares):
    return [t for w,t in pares]

#########################################################
T=[]
for i in range(0,len(Textos)):
    T.append(re.sub(',',' ', Textos[i]))

#print(T[0]);exit(1)

TAGS=[]
for i in range(0,len(T)):
    t=T[i]
    t=t.decode("utf-8")
    sents=tok.tokenize(t)
    m = a.TAGGER2
    h = a.anota_sentencas([sents], m)
    TAGS.append(extrai_tags(h[0]))

#print(TAGS[0]);exit(1)

ndifpostag=[]
for i in range(0,len(TAGS)):ndifpostag.append(len(set(TAGS[i])))

#print(len(ndifpostag))
#print(ndifpostag);exit(1)

SR=[]
for i in range(0,len(TAGS)):SR.append(TAGS[i].count('SR')+
                                      TAGS[i].count('SR-F')+
                                      TAGS[i].count('SR-I')+
                                      TAGS[i].count('SR-P')+
                                      TAGS[i].count('SR-SP')+
                                      TAGS[i].count('SR-D')+
                                      TAGS[i].count('SR-RA')+
                                      TAGS[i].count('SR-SD')+
                                      TAGS[i].count('SR-R')+
                                      TAGS[i].count('SR-SR')+
                                      TAGS[i].count('SR-G')+
                                      TAGS[i].count('SR-PP'))

#print(len(SR))
#print(SR);exit(1)

HV=[]
for i in range(0,len(TAGS)):HV.append(TAGS[i].count('HV')+
                                      TAGS[i].count('HV-F')+
                                      TAGS[i].count('HV-I')+
                                      TAGS[i].count('HV-P')+
                                      TAGS[i].count('HV-SP')+
                                      TAGS[i].count('HV-D')+
                                      TAGS[i].count('HV-RA')+
                                      TAGS[i].count('HV-SD')+
                                      TAGS[i].count('HV-R')+
                                      TAGS[i].count('HV-SR')+
                                      TAGS[i].count('HV-G')+
                                      TAGS[i].count('HV-PP')+
                                      TAGS[i].count('HV-NA'))

#print(len(HV))
#print(HV);exit(1)

ET=[]
for i in range(0,len(TAGS)):ET.append(TAGS[i].count('ET')+
                                      TAGS[i].count('ET-F')+
                                      TAGS[i].count('ET-I')+
                                      TAGS[i].count('ET-P')+
                                      TAGS[i].count('ET-SP')+
                                      TAGS[i].count('ET-D')+
                                      TAGS[i].count('ET-RA')+
                                      TAGS[i].count('ET-SD')+
                                      TAGS[i].count('ET-R')+
                                      TAGS[i].count('ET-SR')+
                                      TAGS[i].count('ET-G')+
                                      TAGS[i].count('ET-PP'))

#print(len(ET))
#print(ET);exit(1)

TR=[]
for i in range(0,len(TAGS)):TR.append(TAGS[i].count('TR')+
                                      TAGS[i].count('TR-F')+
                                      TAGS[i].count('TR-I')+
                                      TAGS[i].count('TR-P')+
                                      TAGS[i].count('TR-SP')+
                                      TAGS[i].count('TR-D')+
                                      TAGS[i].count('TR-RA')+
                                      TAGS[i].count('TR-SD')+
                                      TAGS[i].count('TR-R')+
                                      TAGS[i].count('TR-SR')+
                                      TAGS[i].count('TR-G')+
                                      TAGS[i].count('TR-PP')+
                                      TAGS[i].count('TR-NA'))

#print(len(TR))
#print(TR);exit(1)

VB=[]
for i in range(0,len(TAGS)):VB.append(TAGS[i].count('VB')+
                                      TAGS[i].count('VB-F')+
                                      TAGS[i].count('VB-I')+
                                      TAGS[i].count('VB-P')+
                                      TAGS[i].count('VB-SP')+
                                      TAGS[i].count('VB-D')+
                                      TAGS[i].count('VB-RA')+
                                      TAGS[i].count('VB-SD')+
                                      TAGS[i].count('VB-R')+
                                      TAGS[i].count('VB-SR')+
                                      TAGS[i].count('VB-G')+
                                      TAGS[i].count('VB-PP')+
                                      TAGS[i].count('VB-AN'))

#print(len(VB))
#print(VB);exit(1)

AG=[]
for i in range(0,len(TAGS)):AG.append(TAGS[i].count('-F')+
                                      TAGS[i].count('-G')+
                                      TAGS[i].count('-P'))

#print(len(AG))
#print(AG);exit(1)

N=[]
for i in range(0,len(TAGS)):N.append(TAGS[i].count('N')+
                                     TAGS[i].count('N-P')+
                                     TAGS[i].count('NPR')+
                                     TAGS[i].count('NPR-P'))

#print(len(N))
#print(N);exit(1)

PRO=[]
for i in range(0,len(TAGS)):PRO.append(TAGS[i].count('PRO')+
                                       TAGS[i].count('P+PRO')+
                                       TAGS[i].count('PRO$')+
                                       TAGS[i].count('PRO$-F')+
                                       TAGS[i].count('PRO$-P')+
                                       TAGS[i].count('PRO$-F-P'))

#print(len(PRO))
#print(PRO);exit(1)

CL=[]
for i in range(0,len(TAGS)):CL.append(TAGS[i].count('CL')+
                                       TAGS[i].count('CL+CL')+
                                       TAGS[i].count('...+CL')+
                                       TAGS[i].count('...+CL+CL')+
                                       TAGS[i].count('SR-R!CL')+
                                       TAGS[i].count('ET-R!CL')+
                                       TAGS[i].count('HV-R!CL')+
                                       TAGS[i].count('TR-R!CL')+
                                       TAGS[i].count('VB-R!CL'))

#print(len(CL))
#print(CL);exit(1)

D=[]
for i in range(0,len(TAGS)):D.append(TAGS[i].count('D')+
                                       TAGS[i].count('D-F')+
                                       TAGS[i].count('D-P')+
                                       TAGS[i].count('D-F-P')+
                                       TAGS[i].count('D-G')+
                                       TAGS[i].count('D-G-P')+
                                       TAGS[i].count('D-UM')+
                                       TAGS[i].count('D-UM-F')+
                                       TAGS[i].count('D-UM-P')+
                                       TAGS[i].count('D-UM-F-P')+
                                       TAGS[i].count('DEM'))

#print(len(D))
#print(D);exit(1)

ADJ=[]
for i in range(0,len(TAGS)):ADJ.append(TAGS[i].count('ADJ')+
                                       TAGS[i].count('ADJ-F')+
                                       TAGS[i].count('ADJ-G')+
                                       TAGS[i].count('ADJ-P')+
                                       TAGS[i].count('ADJ-F-P')+
                                       TAGS[i].count('ADJ-G-P')+
                                       TAGS[i].count('ADJ-R')+
                                       TAGS[i].count('ADJ-R-F')+
                                       TAGS[i].count('ADJ-R-P')+
                                       TAGS[i].count('ADJ-R-F-P')+
                                       TAGS[i].count('ADJ-R-G')+
                                       TAGS[i].count('ADJ-R-G-P')+
                                       TAGS[i].count('ADJ-S')+
                                       TAGS[i].count('ADJ-S-F')+
                                       TAGS[i].count('ADJ-S-P')+
                                       TAGS[i].count('ADJ-S-F-P'))

#print(len(ADJ))
#print(ADJ);exit(1)

ADV=[]
for i in range(0,len(TAGS)):ADV.append(TAGS[i].count('ADV')+
                                       TAGS[i].count('ADV-R')+
                                       TAGS[i].count('ADV-S')+
                                       TAGS[i].count('.../P.../N')+
                                       TAGS[i].count('.../P.../ADJ')+
                                       TAGS[i].count('.../P.../ADV')+
                                       TAGS[i].count('.../ADV.../P')+
                                       TAGS[i].count('ADV-NEG'))

#print(len(ADV))
#print(ADV);exit(1)

Q=[]
for i in range(0,len(TAGS)):Q.append(TAGS[i].count('Q')+
                                       TAGS[i].count('Q-F')+
                                       TAGS[i].count('Q-P')+
                                       TAGS[i].count('Q-F-P')+
                                       TAGS[i].count('Q-G')+
                                       TAGS[i].count('Q-G-P')+
                                       TAGS[i].count('Q-NEG')+
                                       TAGS[i].count('Q-NEG-P')+
                                       TAGS[i].count('Q-NEG-F')+
                                       TAGS[i].count('Q-NEG-F-P'))

#print(len(Q))
#print(Q);exit(1)

CONJ=[]
for i in range(0,len(TAGS)):CONJ.append(TAGS[i].count('CONJ')+
                                       TAGS[i].count('CONJ-NEG')+
                                       TAGS[i].count('C')+
                                       TAGS[i].count('Various'))

#print(len(CONJ))
#print(CONJ);exit(1)

WPRO=[]
for i in range(0,len(TAGS)):WPRO.append(TAGS[i].count('WPRO')+
                                        TAGS[i].count('WPRO-P')+
                                        TAGS[i].count('WPRO-F-P')+
                                        TAGS[i].count('WPRO$')+
                                        TAGS[i].count('WPRO$-F')+
                                        TAGS[i].count('WPRO$-P')+
                                        TAGS[i].count('WPRO$-F-P')+
                                        TAGS[i].count('WQ')+
                                        TAGS[i].count('WD')+
                                        TAGS[i].count('WD-F')+
                                        TAGS[i].count('WD-P')+
                                        TAGS[i].count('WD-F-P')+
                                        TAGS[i].count('WADV'))

#print(len(WPRO))
#print(WPRO);exit(1)

P=[]
for i in range(0,len(TAGS)):P.append(TAGS[i].count('P')+
                                     TAGS[i].count('P+D')+
                                     TAGS[i].count('P+D-P')+
                                     TAGS[i].count('P+D-F')+
                                     TAGS[i].count('P+D-F-P')+
                                     TAGS[i].count('P+D-UM')+
                                     TAGS[i].count('P+D-UM-P')+
                                     TAGS[i].count('P+D-UM-F')+
                                     TAGS[i].count('P+D-UM-F-P')+
                                     TAGS[i].count('P+PRO')+
                                     TAGS[i].count('P+OUTRO')+
                                     TAGS[i].count('P+OUTRO-P')+
                                     TAGS[i].count('P+OUTRO-F')+
                                     TAGS[i].count('P+OUTRO-F-P')+
                                     TAGS[i].count('P+Q')+
                                     TAGS[i].count('P+Q-P')+
                                     TAGS[i].count('P+Q-F')+
                                     TAGS[i].count('P+Q-F-P')+
                                     TAGS[i].count('P+WPRO')+
                                     TAGS[i].count('P+DEM')+
                                     TAGS[i].count('P+ADV')+
                                     TAGS[i].count('P+WADV')+
                                     TAGS[i].count('P+CL')+
                                     TAGS[i].count('P+NPR'))

#print(len(P))
#print(P);exit(1)

CLU=[]
for i in range(0,len(TAGS)):CLU.append(TAGS[i].count('P...P')+
                                        TAGS[i].count('ADV...P')+
                                        TAGS[i].count('P+D...ADV')+
                                        TAGS[i].count('NEG...ADJ-G'))

#print(len(CLU))
#print(CLU);exit(1)

OUTRO=[]
for i in range(0,len(TAGS)):OUTRO.append(TAGS[i].count('OUTRO')+
                                        TAGS[i].count('OUTRO-P')+
                                        TAGS[i].count('OUTRO-F')+
                                        TAGS[i].count('OUTRO-F-P'))

#print(len(OUTRO))
#print(OUTRO);exit(1)

FP=[]
for i in range(0,len(TAGS)):FP.append(TAGS[i].count('FP'))

#print(len(FP))
#print(FP);exit(1)

NUM=[]
for i in range(0,len(TAGS)):NUM.append(TAGS[i].count('NUM')+
                                       TAGS[i].count('NUM-F'))

#print(len(NUM))
#print(NUM);exit(1)

NEG=[]
for i in range(0,len(TAGS)):NEG.append(TAGS[i].count('NEG')+
                                       TAGS[i].count('SENAO')+
                                       TAGS[i].count('CONJ-NEG')+
                                       TAGS[i].count('ADV-NEG')+
                                       TAGS[i].count('Q-NEG')+
                                       TAGS[i].count('Q-NEG-P')+
                                       TAGS[i].count('Q-NEG-F')+
                                       TAGS[i].count('Q-NEG-F-P'))

#print(len(NEG))
#print(NEG);exit(1)

INTJ=[]
for i in range(0,len(TAGS)):INTJ.append(TAGS[i].count('INTJ'))

#print(len(INTJ))
#print(INTJ);exit(1)

FW=[]
for i in range(0,len(TAGS)):FW.append(TAGS[i].count('FW'))

#print(len(FW))
#print(FW);exit(1)

XX=[]
for i in range(0,len(TAGS)):XX.append(TAGS[i].count('XX'))

#print(len(XX))
#print(XX);exit(1)

L=[ndifpostag,SR,HV,ET,TR,VB,N,PRO,CL,D,ADJ,ADV,Q,CONJ,WPRO,
   P,OUTRO,FP,NUM,NEG,INTJ,NotasF,TemaF,CoerenciaF,RegrasF]

L=list(zip(*L))
H=[]
for i in range(len(L)):H.append(list(L[i]))

df = pd.DataFrame(H,columns=['ndifpostag','SR','HV','ET',
                             'TR','VB','N','PRO','CL','D',
                             'ADJ','ADV','Q','CONJ','WPRO',
                             'P','OUTRO','FP','NUM','NEG',
                             'INTJ','NotasF','TemaF','CoerenciaF','RegrasF'])

df.to_csv('sintatica.csv', sep=';', encoding='utf-8')
