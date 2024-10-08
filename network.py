import math
import random

class network:
  def __init__(self,hlayers,neurons,inputs,outputs,learningrate=5e-5): #num of hidden layers, neurons in each hidden layer, number of inputs, number of outputs, and learning rate (default value set to 5e-5)
    self.network = {}
    self.results = {}
    self.inputs = inputs #number of input values
    self.outputs = outputs #number of output values
    self.hlayers = hlayers #number of hidden layers
    self.neurons = neurons #number of neurons per hidden layer
    self.LearningRate = learningrate
    IPL = {} #stands for input layer
    for i in range(inputs):
      IPL[str(i)] = 0.0
    self.network["inputlayer"] = IPL
    self.network["inputlayer"]["keys"] = list(IPL.keys())
    self.network["hiddenlayers"] = {}
    self.network["hiddenlayers"]["results"] = {}
    self.network["hiddenlayers"]["keys"] = []
    self.network["hiddenlayers"]["weights"] = []
    HLC = {} #stands for hidden layer content
    for i in range(hlayers):
      HLC[str(i)] = {}
      HLK = [] #stands for hidden layer keys
      for j in range(neurons):
        index = str(j+(i*neurons)+inputs)
        HLK.append(index)
        HLC[str(i)][str(index)] = 0.0
      self.network["hiddenlayers"]["results"] = HLC.copy()
      self.network["hiddenlayers"]["keys"].append(HLK)
    FHLW = {} #stands for first hidden layer weights
    for i in self.network["hiddenlayers"]["keys"][0]:
      FHLW[i] = {}
      for j in self.network["inputlayer"]["keys"]:
        FHLW[i][j] = random.uniform(0.000,1.000)
    self.network["hiddenlayers"]["weights"].append(FHLW)
    for i in range(1,len(self.network["hiddenlayers"]["keys"])):
      HLW = {} #stands for hidden layer weights
      CLW  = {} #stands for current layer weights
      currentkeys = self.network["hiddenlayers"]["keys"][i]
      previouslayer = self.network["hiddenlayers"]["keys"][i-1]
      for j in previouslayer:
        CLW[j] = random.uniform(0.000,1.000)
      for k in currentkeys:
        HLW[k] = CLW.copy()
      self.network["hiddenlayers"]["weights"].append(HLW.copy())
    self.network["outputlayer"] = {}
    OPL = {} #stands for output layer (results)
    OPLW = {} #stands for output layer weights
    previouslayer = self.network["hiddenlayers"]["keys"][-1]
    for i in range(outputs):
      OPLW[str(i+(hlayers*neurons)+self.inputs)] = {}
      for j in previouslayer:
        OPLW[str(i+(hlayers*neurons)+self.inputs)][j] = random.uniform(0.000,1.000)
      OPL[str(i+(hlayers*neurons)+self.inputs)] = 0.0
    self.network["outputlayer"]["weights"] = OPLW
    self.network["outputlayer"]["results"] = OPL
    self.network["outputlayer"]["keys"] = list(OPL.keys())

  def __str__(self):
    structurevisual = ""
    for i in self.network["inputlayer"]["keys"]:
      structurevisual += "Input #"
      structurevisual += i
      structurevisual += ":"
      structurevisual += str(self.network["inputlayer"][i])
      structurevisual += "\n"
    structurevisual += "\n\n"
    for i in self.network["hiddenlayers"]["results"].keys():
      structurevisual += "Hidden Layer #"
      structurevisual += i
      structurevisual += ":"
      structurevisual += str(self.network["hiddenlayers"]["results"][i])
      structurevisual +="\n"
    structurevisual += "\n\n"
    for i in self.network["outputlayer"]["keys"]:
      structurevisual += "Output:Neuron #"
      structurevisual += i
      structurevisual += ":"
      structurevisual += str(self.network["outputlayer"]["weights"][i])
      structurevisual += "\n"
    structurevisual += "\nFinal Result Values: \n"
    for i in self.results:
      structurevisual += str(i)
      structurevisual += "\n"
    structurevisual += "\n\n"
    return structurevisual

  def LogisticFunction(self,x):
    val = 1/(1+math.e**(-0.0639*(x-0.5)))
    return val

  def SetInput(self,ips): #stands for inputs
    try:
      for i in range(self.inputs):
        index = str(i)
        try:
          ips[i] = float(ips[i])
        except ValueError:
          return f"input at index {i} is not a numerical value."
        self.network["inputlayer"][index] = self.LogisticFunction(ips[i])
    except IndexError:
      return "given inputs does not match set number of input values in network."

  def GetLayers(self):
    return(self.hlayers)

  def GetLayerState(self):
    structurevisual = ""
    for i in self.network["hiddenlayers"]["results"].keys():
      structurevisual += str(i)
      structurevisual += ": "
      structurevisual += str(self.network["hiddenlayers"]["results"][i])
      structurevisual +="\n"
    index = 0
    for i in self.network["hiddenlayers"]["weights"]:
      structurevisual += "hidden layer #"
      structurevisual += str(index)
      structurevisual += ": \n"
      for j in i.keys():
        structurevisual += j
        structurevisual += ":"
        structurevisual +="\n"
        for k in i[j].keys():
          structurevisual += k
          structurevisual += ":"
          structurevisual += str(i[j][k])
          structurevisual +="\n"
      structurevisual += "\n"
      index +=1
    return structurevisual

  def IterateFirstHiddenLayer(self):
    for i in self.network["hiddenlayers"]["keys"][0]:
      for j in self.network["inputlayer"]["keys"]:
        self.network["hiddenlayers"]["results"]["0"][i] += self.network["inputlayer"][j]*self.network["hiddenlayers"]["weights"][0][i][j]
      self.network["hiddenlayers"]["results"]["0"][i] = self.LogisticFunction(self.network["hiddenlayers"]["results"]["0"][i])

  def IterateOtherHiddenLayers(self):
    for i in range(1,self.hlayers):
      for j in self.network["hiddenlayers"]["keys"][i]:
        for k in self.network["hiddenlayers"]["keys"][i-1]:
          self.network["hiddenlayers"]["results"][str(i)][j] += self.network["hiddenlayers"]["results"][str(i-1)][k]*self.network["hiddenlayers"]["weights"][i][str(j)][k]
        self.network["hiddenlayers"]["results"][str(i)][j] = self.LogisticFunction(self.network["hiddenlayers"]["results"][str(i)][j])

  def IterateAllHiddenLayers(self):
    self.IterateFirstHiddenLayer()
    if self.hlayers > 1:
      self.IterateOtherHiddenLayers()

  def IterateResultLayer(self):
    for i in self.network["outputlayer"]["keys"]:
      for j in self.network["hiddenlayers"]["keys"][-1]:
        self.network["outputlayer"]["results"][i] += self.network["hiddenlayers"]["results"][list(self.network["hiddenlayers"]["results"].keys())[-1]][j]*self.network["outputlayer"]["weights"][i][j]
      self.network["outputlayer"]["results"][i] = self.LogisticFunction(self.network["outputlayer"]["results"][i])
    self.results = list(self.network["outputlayer"]["results"].values())

  def IterateAllLayers(self,inputs):
    self.SetInput(inputs)
    self.IterateAllHiddenLayers()
    self.IterateResultLayer()

  def UpdateWeight(self,key1,key2,rate=None, customvalue =None):
    locations = ["",""]
    index = [0,0]
    keys = [key1,key2]
    if not rate:
      if not customvalue:
        rate = self.LearningRate

    for i in keys:
      for j in self.network["hiddenlayers"]["keys"]:
        if i in j:
          index[keys.index(i)] = self.network["hiddenlayers"]["keys"].index(j)
          locations[keys.index(i)] = "hiddenlayers"
      if locations[keys.index(i)] == "":
        if i in self.network["outputlayer"]["keys"]:
          locations[keys.index(i)] = "outputlayer"
        elif i in self.network["inputlayer"]["keys"] and keys.index(i) == 1:
          locations[keys.index(i)] = "inputlayer"
        else:
          return "key not found"
    if not customvalue:
      if locations[0] == "hiddenlayers":
        self.network[locations[0]]["weights"][index[0]][key1][key2] += rate
      elif locations[0] == "outputlayer":
        self.network[locations[0]]["weights"][key1][key2] += rate 
    else:
      if locations[0] == "hiddenlayers":
        self.network[locations[0]]["weights"][index[0]][key1][key2] = customvalue
      elif locations[0] == "outputlayer":
        self.network[locations[0]]["weights"][key1][key2] = customvalue


