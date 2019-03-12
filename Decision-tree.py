import matplotlib.pyplot as plt
from math import log
import operator



# 计算给定数据集的香农熵（根据dataSet中所有的特征向量的类别计算熵） 
# dataSet：给定数据集 
# 返回shannonEnt：香农熵 
import numpy
def calcShannonEnt(dataSet): 
    numEntries = len(dataSet) 
    # 创建一个空字典 
    labelCounts = {} 
    # for循环：使labelCounts字典保存多个键值对，并且以dataSet中数据的类别（标签）为键，该类别数据的条数为对应的值 
    for featVec in dataSet: 
        currentLabel = featVec[-1] 
        if currentLabel not in labelCounts.keys(): 
            # keys()方法返回字典中的键 
            labelCounts[currentLabel] = 0 
            # 如果labelCounts中没有currentLabel，则添加一个以currentLabel为键的键值对，值为0 
        labelCounts[currentLabel] += 1 
        # 将labelCounts中类型为currentLabel值加1
    shannonEnt = 0.0 
    for key in labelCounts: 
        # 根据熵的公式进行累加 
        prob = float(labelCounts[key])/numEntries 
        # 计算每种数据类别出现的概率 
        shannonEnt -= prob * log(prob, 2) 
        # 根据定义的公式计算信息 
    return shannonEnt

# 按照给定特征划分数据集 
# dataSet：给定数据集 
# axis：给定特征所在特征向量的列 
# value：给定特征的特征值
# 返回retDataSet：划分后的数据集 
def splitDataSet(dataSet, axis, value): 
    retDataSet = [] 
    for featVec in dataSet: 
        if featVec[axis] == value: 
            # 若当前特征向量指定特征列（第axis列，列从0开始）的特征值与给定的特征值（value）相等 
            # 下面两行代码相当于将axis列去掉 
            reducedFeatVec = featVec[:axis] 
            # 取当前特征向量axis列之前的列的特征 
            reducedFeatVec.extend(featVec[axis+1:]) 
            # 将上一句代码取得的特征向量又加上axis列后的特征 
            retDataSet.append(reducedFeatVec) 
            # 将划分后的特征向量添加到retDataSet中
    return retDataSet

# 选择最好的数据划分方式 
# dataSet：要进行划分的数据集 
# 返回bestFeature：在分类时起决定性作用的特征（下标） 
def chooseBestFeatureToSplit(dataSet): 
    numFeatures = len(dataSet[0]) - 1 
    # 特征的数量 
    baseEntropy = calcShannonEnt(dataSet) 
    # 计算数据集的香农熵 
    bestInfoGain = 0.0 
    # bestInfoGain=0：最好的信息增益为0，表示再怎么划分， 
    # 香农熵（信息熵）都不会再变化，这就是划分的最优情况 
    bestFeature = -1 
    for i in range(numFeatures): 
        # 根据数据的每个特征进行划分，并计算熵， 
        # 熵减少最多的情况为最优，此时对数据进行划分的特征作为划分的最优特征 
        featList = [example[i] for example in dataSet]
        # featList为第i列数据（即第i个特征的所有特征值的列表（有重复）） 
        uniqueVals = set(featList) 
        # uniqueVals为第i列特征的特征值（不重复，例如有特征值1,1,0,0，uniqueVals为[0, 1]） 
        newEntropy = 0.0 
        for value in uniqueVals: 
            subDataSet = splitDataSet(dataSet, i, value) 
            prob = len(subDataSet)/float(len(dataSet)) 
            newEntropy += prob * calcShannonEnt(subDataSet) 
            # newEntropy为将数据集根据第i列特征进行划分的 
            # 所有子集的熵乘以该子集占总数据集比例的和 
            infoGain = baseEntropy - newEntropy 
            # 计算信息增益，即熵减 
            if infoGain > bestInfoGain: 
                bestInfoGain = infoGain 
                bestFeature = i 
    return bestFeature

# 当数据集已经处理了所有属性，但是分类标签依然不唯一时，采用多数表决的方法决定叶子结点的分类 
def majorityCnt(classList): 
    classCount = {} 
    for vote in classList: 
        if vote not in classCount.keys(): 
            classCount[vote] = 0 
            classCount[vote] += 1 
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]

# 利用函数递归创建决策树 
# dataSet：数据集
# labels：标签列表，包含了数据集中所有特征的标签 
def createTree(dataSet, labels): 
    classList = [example[-1] for example in dataSet] # 取出dataSet最后一列的数据 
    if classList.count(classList[0]) == len(classList): 
        # classList中classList[0]出现的次数=classList长度，表示类别完全相同，停止继续划分 
        return classList[0] 
    if len(dataSet[0]) == 1: 
        # 遍历完所有特征时返回出现次数最多的类别 
        return majorityCnt(classList) 
    bestFeat = chooseBestFeatureToSplit(dataSet) 
    # 计算划分的最优特征（下标） 
    bestFeatLabel = labels[bestFeat] # 数据划分的最优特征的标签（即是什么特征） 
    myTree = {bestFeatLabel:{}} # 创建一个树（字典），bestFeatLabel为根结点 
    del(labels[bestFeat]) 
    featValues = [example[bestFeat] for example in dataSet] 
    uniqueVals = set(featValues) 
    for value in uniqueVals: 
        subLabels = labels[:] 
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)# 利用递归构造决策树 
    return myTree

import matplotlib.pyplot as plt # 定义文本框和箭头格式 
decisionNode = dict(boxstyle="sawtooth", fc="0.8") 
leafNode = dict(boxstyle="round4", fc="0.8") 
arrow_args = dict(arrowstyle="<-") # 使用文本注解绘制树节点 

def plotNode(nodeText, centerPt, parentPt, nodeType): 
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', 
                                va="center", ha="center", bbox=nodeType, arrowprops=arrow_args) 

def createPlot(): 
    fig = plt.figure(1, facecolor='white') 
    fig.clf() 
    createPlot.ax1 = plt.subplot(111, frameon=False) 
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode) 
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode) 
    plt.show()

# 获取叶节点的数目，以便确定x轴的长度 
def getNumLeafs(myTree): 
    numLeafs = 0 
    firstStr = list(myTree.keys())[0] #根结点 
    secondDict = myTree[firstStr] 
    for key in secondDict.keys(): 
        if type(secondDict[key]).__name__ == 'dict': 
            numLeafs += getNumLeafs(secondDict[key]) 
        else: 
            numLeafs += 1 
    return numLeafs # 获取决策树的深度 

def getTreeDepth(myTree): 
    maxDepth = 0 
    firstStr = list(myTree.keys())[0] 
    secondDict = myTree[firstStr] 
    for key in secondDict.keys(): 
        if type(secondDict[key]).__name__ == 'dict': 
            thisDepth = 1 + getTreeDepth(secondDict[key]) 
        else: 
            thisDepth = 1 
        if thisDepth > maxDepth: 
            maxDepth = thisDepth 
    return maxDepth

# 在父子节点间填充文本信息 
def plotMidText(cntrPt, parentPt, textString): 
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0] 
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1] 
    createPlot.ax1.text(xMid, yMid, textString) 
    
def plotTree(myTree, parentPt, nodeText): 
    numLeafs = getNumLeafs(myTree) 
    depth = getTreeDepth(myTree) 
    firstStr = list(myTree.keys())[0] 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff) 
    plotMidText(cntrPt, parentPt, nodeText) 
    plotNode(firstStr, cntrPt, parentPt, decisionNode) 
    secondDict = myTree[firstStr] 
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD 
    for key in secondDict.keys(): 
        if type(secondDict[key]).__name__ == 'dict': 
            plotTree(secondDict[key], cntrPt, str(key)) 
        else: 
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW 
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) 
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key)) 
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD 
            
def createPlot(inTree): 
    fig = plt.figure(1, facecolor='white') 
    fig.clf() 
    axprops = dict(xticks=[], yticks=[]) 
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) 
    plotTree.totalW = float(getNumLeafs(inTree)) 
    plotTree.totalD = float(getTreeDepth(inTree)) 
    plotTree.xOff = -0.5/plotTree.totalW; 
    plotTree.yOff = 1.0 
    plotTree(inTree, (0.5, 1.0), '') 
    plt.show()

# 使用决策树的分类函数 
def classify(inputTree, featLabels, testVec): 
    firstStr = list(inputTree.keys())[0] 
    secondDict = inputTree[firstStr] 
    featIndex = featLabels.index(firstStr) 
    for key in secondDict.keys(): 
        if testVec[featIndex] == key: 
            if type(secondDict[key]).__name__ == 'dict': 
                classLabel = classify(secondDict[key], featLabels, testVec) 
            else: 
                classLabel = secondDict[key] 
    return classLabel

# 使用pickle模块存储决策树 
def storeTree(inputTree, filename): 
    import pickle 
    fw = open(filename, 'w') 
    pickle.dump(inputTree, fw) 
    fw.close() 
    
def grabTree(filename): 
    import pickle 
    fr = open(filename) 
    return pickle.load(fr)

# 使用lenses.txt中的数据构造决策树 
#fr = open('lenses.txt') 
fr = open('E:\\data\\machinelearninginaction\\Ch03\\lenses.txt') 
lenses = [inst.strip().split('\t') for inst in fr.readlines()] 
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate'] 
lensesTree = createTree(lenses, lensesLabels) # 画决策树 
createPlot(lensesTree)




