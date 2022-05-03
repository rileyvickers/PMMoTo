import numpy as np

def readPorousMediaXYZR(domainFile):
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
    sphereData[3,:] = rSphere*rSphere

    return domainDim,sphereData
