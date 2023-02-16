import numpy as np
import gzip 

def readPorousMediaXYZR(file):
    
    domainFile = open(file, 'r')
    Lines = domainFile.readlines()
    numObjects = len(Lines) - 3

    xSphere = np.zeros(numObjects)
    ySphere = np.zeros(numObjects)
    zSphere = np.zeros(numObjects)
    rSphere = np.zeros(numObjects)
    c = 0
    c2 = 0
    for line in Lines:
        if(c==0):
            xMin = float(line.split(" ")[0])
            xMax = float(line.split(" ")[1])
        elif(c==1):
            yMin = float(line.split(" ")[0])
            yMax = float(line.split(" ")[1])
        elif(c==2):
            zMin = float(line.split(" ")[0])
            zMax = float(line.split(" ")[1])
        elif(c>2):
            try:
                xSphere[c2] = float(line.split(" ")[0])
                ySphere[c2] = float(line.split(" ")[1])
                zSphere[c2] = float(line.split(" ")[2])
                rSphere[c2] = float(line.split(" ")[3])
            except ValueError:
                xSphere[c2] = float(line.split("\t")[0])
                ySphere[c2] = float(line.split("\t")[1])
                zSphere[c2] = float(line.split("\t")[2])
                rSphere[c2] = float(line.split("\t")[3])
            c2 = c2 + 1
        c = c + 1


    domainDim = np.array([[xMin,xMax],[yMin,yMax],[zMin,zMax]])
    sphereData = np.zeros([4,numObjects])
    sphereData[0,:] = xSphere
    sphereData[1,:] = ySphere
    sphereData[2,:] = zSphere
    sphereData[3,:] = rSphere
    domainFile.close()

    return domainDim,sphereData

def readPorousMediaLammpsDump(file, sigmaLJ=[0.25]):
    
    indexVars = ['x', 'y', 'z', 'type']
    indexes = []

    if file.endswith('.gz'):
        domainFile = gzip.open(file,'rt')
    else:
        domainFile = open(file,'r')

    Lines = domainFile.readlines()

    for i in range(9):
        preamLine = Lines.pop(0)
        if (i == 1):
            timeStep = preamLine
        elif (i == 3):
            numObjects = int(preamLine)
        elif (i == 5):
            xMin = float(preamLine.split(" ")[0])
            xMax = float(preamLine.split(" ")[1])
        elif (i == 6):
            yMin = float(preamLine.split(" ")[0])
            yMax = float(preamLine.split(" ")[1])
        elif (i == 7):
            zMin = float(preamLine.split(" ")[0])
            zMax = float(preamLine.split(" ")[1])
        elif (i == 8):
            colTypes = [j.strip() for j in preamLine.split(" ")[2:]]

    for var in indexVars:
        indexes.append(colTypes.index(var))  
        
    xSphere = np.zeros(numObjects)
    ySphere = np.zeros(numObjects)
    zSphere = np.zeros(numObjects)
    rSphere = np.zeros(numObjects)

    c = 0
    for line in Lines:
        xSphere[c] = float(line.split(" ")[indexes[0]])
        ySphere[c] = float(line.split(" ")[indexes[1]])
        zSphere[c] = float(line.split(" ")[indexes[2]])
        rSphere[c] = float(sigmaLJ[int(line.split(" ")[indexes[3]])-1])
        c += 1

    domainDim = np.array([[xMin, xMax],[yMin, yMax],[zMin, zMax]])
    sphereData = np.zeros([4, numObjects])
    sphereData[0,:] = xSphere
    sphereData[1,:] = ySphere
    sphereData[2,:] = zSphere
    sphereData[3,:] = rSphere
    domainFile.close()

    return domainDim, sphereData