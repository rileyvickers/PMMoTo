from .subDomain import genDomainSubDomain
from .distance import calcEDT
from .dataRead import readPorousMediaXYZR
from .drainage import calcDrainage
from .morphology import morph
import communication
from mpi4py import MPI
import numpy as np
import edt
