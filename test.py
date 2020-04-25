from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
import collections

def nameentropy(namestring):
    counter_char = collections.Counter(namestring)
    entropy = 0
    for c, ctn in counter_char.items():
        _p = float(ctn)/len(namestring)
        entropy += -1 * _p * math.log(_p, 2)
    return round(entropy, 7)

def numberssum(namestring):
    digsum=0
    for strs in namestring:
        if strs.isdigit():
            digsum+=1
    return digsum

domainlist = []
testdomainlist=[]
#fealist=[]
class Domain:
    def __init__(self,_name,_label, _namelength, _numbers, _entropy):
        self.name = _name
        self.label = _label
        self.namelength = _namelength
        self.numbers = _numbers
        self.entropy = _entropy

    def returnData(self):
        return [self.namelength, self.numbers, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return "dga"
        else:
            return "notdga"

def inittrainData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            namelength = len(name)
            numbers = nameentropy(name)
            entropy =nameentropy(name)
            domainlist.append(Domain(name,label,namelength,numbers,entropy))

def inittestData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            name = tokens[0]
            lable=''
            namelength = len(name)
            numbers = nameentropy(name)
            entropy =nameentropy(name)
            testdomainlist.append(Domain(name,lable,namelength,numbers,entropy))


resultfile=open("result.txt","a+")
inittrainData("train.txt")
inittestData("test.txt")
featureMatrix = []
labelList = []
for item in domainlist:
    featureMatrix.append(item.returnData())
    labelList.append(item.returnLabel())
clf = RandomForestClassifier(random_state=0)
clf.fit(featureMatrix,labelList)
for testitem in testdomainlist:
    fealist=[testitem.namelength,testitem.numbers,testitem.entropy]
    taglable=clf.predict(fealist)
    tag=str(taglable)
    tag1=tag.replace("'","")
    tag2=tag1.replace("[","")
    tag3=tag2.replace("]","")
    resultfile.write(testitem.name)
    resultfile.write(",")
    resultfile.write(tag3)
    resultfile.write("\n")

resultfile.close()




