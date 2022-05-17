import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

def subDomainComm(Orientation,subDomain,sendData):

    comm.Barrier()

    #### FACE ####
    reqs = [None]*Orientation.numFaces
    reqr = [None]*Orientation.numFaces
    recvDataFace = [None]*Orientation.numFaces
    for fIndex in Orientation.faces:
        neigh = subDomain.neighborF[fIndex]
        oppIndex = Orientation.faces[fIndex]['oppIndex']
        oppNeigh = subDomain.neighborF[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[fIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[fIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for fIndex in Orientation.faces:
        orientID = Orientation.faces[fIndex]['ID']
        neigh = subDomain.neighborF[fIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataFace[fIndex] = reqr[fIndex]#.wait()

    #### EDGES ####
    reqs = [None]*Orientation.numEdges
    reqr = [None]*Orientation.numEdges
    recvDataEdge = [None]*Orientation.numEdges
    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[eIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[eIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataEdge[eIndex] = reqr[eIndex]#.wait()

    #### CORNERS ####
    reqs = [None]*Orientation.numCorners
    reqr = [None]*Orientation.numCorners
    recvDataCorner = [None]*Orientation.numCorners
    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        oppIndex = Orientation.corners[cIndex]['oppIndex']
        oppNeigh = subDomain.neighborC[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[cIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[cIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataCorner[cIndex] = reqr[cIndex]#.wait()

    return recvDataFace,recvDataEdge,recvDataCorner
