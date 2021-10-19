'''
======================================================
FINITE ELEMENT PROGRAM FOR MECHANIC ANALSIS
MODEL : 3 nodes plate element
BY : CHI-TATHON KUPWIWAT

REMARK: developed from prof. yamamoto's
Linear stress analysis using triangular plate elements
tri_pltate_3d.py
======================================================
'''

'''
======================================================
IMPORT PART
======================================================
'''
import numpy as np
import itertools
import math

'''
======================================================
CLASS PART
======================================================
'''
class Load:
    def __init__(self):
        self.name = None
        self.size = [0,0,0,0,0,0] # x,y,z direction tx,ty,tz
    def set_name(self,name):
        self.name = name
    def set_size(self,x,y,z,tx,ty,tz):
        self.size = [x,y,z,tx,ty,tz]
    def __repr__(self):
        return "{0},{1}".format(self.name,self.size)
class Node:
    def __init__(self):
        self.name = None
        self.coord = []
        self.loads = []
        self.res = [0,0,0,0,0,0]
        self.dof = []
        self.global_d =[[0],[0],[0],[0],[0],[0]]
    def set_name(self,name):
        self.name = name
    def set_coord(self,x,y,z):
        self.coord = [x,y,z]
    def set_loads(self,load):
        self.loads.append(load)
    def set_res(self,rx,ry,rz,tx,ty,tz):
        self.res = [rx,ry,rz,tx,ty,tz] #0==no restrain,1== restrained
    def __repr__(self):
        return "{0},{1},{2},{3}".format(self.name,self.coord,self.res,self.loads)

class Element:
    def __init__(self):
        self.name = None
        self.nodes = []
        self.snodes = [] # sorted node
        self.element_dof = []
        self.lcoord = []
        #--------------------------------------
        self.young = 0
        self.poisson = 0
        self.thickness = 0
        #--------------------------------------
        self.area = 0
        self.g_weigth = [27/60,8/60,8/60,8/60,3/60,3/60,3/60]
        self.L1i =      [1/3  ,1/2 ,0   ,1/2 ,1   ,0   ,0   ]
        self.L2i =      [1/3  ,1/2 ,1/2 ,0   ,0   ,1   ,0   ]
        self.L3i =      [1/3  ,0   ,1/2 ,1/2 ,0   ,0   ,1   ]
        #--------------------------------------
        # K of each node
        self.K = None
        #--------------------------------------
        # Transformation matirix
        self.tmatrixsmall = [] #3x3
        self.tmatrix = [] #18x18
        #--------------------------------------
        # obj file
        self.obj_node =[]
        self.obj_element_start = 0
        self.obj_element =[]
    def __repr__(self):
        return "{0},[{1}],[{2}],[{3}],{4},{5},{6}".format(self.name,self.nodes[0],self.nodes[1],self.nodes[2],self.young,self.poisson,self.thickness)


    def gen_obj_node(self,start):
        # transformation matrix 9x9
        self.obj_node = [
        [self.nodes[0].coord[0],self.nodes[0].coord[2],self.nodes[0].coord[1]],
        [self.nodes[1].coord[0],self.nodes[1].coord[2],self.nodes[1].coord[1]],
        [self.nodes[2].coord[0],self.nodes[2].coord[2],self.nodes[2].coord[1]],

        [self.nodes[0].coord[0],self.nodes[0].coord[2]-self.thickness,self.nodes[0].coord[1]],
        [self.nodes[1].coord[0],self.nodes[1].coord[2]-self.thickness,self.nodes[1].coord[1]],
        [self.nodes[2].coord[0],self.nodes[2].coord[2]-self.thickness,self.nodes[2].coord[1]],
        ]

        self.obj_element_start = start

    def gen_obj_element(self):

        num = self.obj_element_start
        self.obj_element = [[1+num,2+num,3+num],[4+num,6+num,5+num],[1+num,3+num,6+num,4+num],[1+num,4+num,5+num,2+num],[3+num,2+num,5+num,6+num]]


    def set_name(self,name):
        self.name = name
    def set_nodes(self,node):
        self.nodes.append(node)
    def set_young(self,value):
        self.young = value
    def set_poisson(self,value):
        self.poisson = value
    def set_thickness(self,value):
        self.thickness = value
    def set_t(self):
        x1,x2,x3 = self.nodes[0].coord[0],self.nodes[1].coord[0],self.nodes[2].coord[0]
        y1,y2,y3 = self.nodes[0].coord[1],self.nodes[1].coord[1],self.nodes[2].coord[1]
        z1,z2,z3 = self.nodes[0].coord[2],self.nodes[1].coord[2],self.nodes[2].coord[2]

        forL1 = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**(0.5)
        forL3 = (((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1))**2 + ((z2-z1)*(x3-x1) - (z3-z1)*(x2-x1))**2 + ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))**2) ** (0.5)

        L11 = (x2-x1)/forL1
        L12 = (y2-y1)/forL1
        L13 = (z2-z1)/forL1
        L31 = ((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1))/forL3
        L32 = ((z2-z1)*(x3-x1) - (z3-z1)*(x2-x1))/forL3
        L33 = ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))/forL3
        L21 = L32*L13 - L12*L33
        L22 = L33*L11 - L13*L31
        L23 = L31*L12 - L11*L32

        self.tmatrixsmall = np.array([[L11,L12,L13],[L21,L22,L23],[L31,L32,L33]])

        self.tmatrix = np.array(
        [[L11,L12,L13, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[L21,L22,L23, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[L31,L32,L33, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 ,L11,L12,L13, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 ,L21,L22,L23, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 ,L31,L32,L33, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 ,L11,L12,L13, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 ,L21,L22,L23, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 ,L31,L32,L33, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L11,L12,L13, 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L21,L22,L23, 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L31,L32,L33, 0 , 0 , 0 , 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L11,L12,L13, 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L21,L22,L23, 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L31,L32,L33, 0 , 0 , 0 ]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L11,L12,L13]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L21,L22,L23]
        ,[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,L31,L32,L33]]
        )

    def _makeArea(self):
        # [rearrage ijk], [make data of ai,bi,ci,area], [make data of li,lxi,lyi]
        ijk = np.array([
            [ 0 , self.nodes[1].coord[0]-self.nodes[0].coord[0] , self.nodes[2].coord[0]-self.nodes[0].coord[0] ],
            [ 0 , self.nodes[1].coord[1]-self.nodes[0].coord[1] , self.nodes[2].coord[1]-self.nodes[0].coord[1] ],
            [ 0 , self.nodes[1].coord[2]-self.nodes[0].coord[2] , self.nodes[2].coord[2]-self.nodes[0].coord[2] ]
            ])

        # local coordinate data * transpose matrix
        ijk = np.dot(self.tmatrixsmall,ijk)
        x1,x2,x3 = ijk[0][0],ijk[0][1],ijk[0][2]
        y1,y2,y3 = ijk[1][0],ijk[1][1],ijk[1][2]

        # data for listA [make data of ai,bi,ci,area]
        ai = [(x2*y3)-(x3*y2),(x3*y1)-(x1*y3),(x1*y2)-(x2*y1)]
        bi = [y2-y3,y3-y1,y1-y2,y2-y3,y3-y1]
        ci = [x3-x2,x1-x3,x2-x1,x3-x2,x1-x3]
        area = 0.5*np.linalg.det(np.array([[1,x1,y1],[1,x2,y2],[1,x3,y3]]))

        # data for listB [make data of li,lxi,lyi]
        li = [  (ai[0]+(bi[0]*ijk[0][0])+(ci[0]*ijk[0][1]))/(2*area),
                (ai[1]+(bi[1]*ijk[0][0])+(ci[1]*ijk[0][1]))/(2*area),
                (ai[2]+(bi[2]*ijk[0][0])+(ci[2]*ijk[0][1]))/(2*area),
                (ai[0]+(bi[0]*ijk[0][0])+(ci[0]*ijk[0][1]))/(2*area),
                (ai[1]+(bi[1]*ijk[0][0])+(ci[1]*ijk[0][1]))/(2*area)]
        lxi = [bi[0]/(2*area),bi[1]/(2*area),bi[2]/(2*area),bi[0]/(2*area),bi[1]/(2*area)]
        lyi = [ci[0]/(2*area),ci[1]/(2*area),ci[2]/(2*area),ci[0]/(2*area),ci[1]/(2*area)]

        return ai,bi,ci,area,li,lxi,lyi

    def _makeDp(self):
        Dp = ((self.young*self.thickness)/(1-(self.poisson**2)))*np.array([[1,self.poisson,0],[self.poisson,1,0],[0,0,(1-self.poisson)/2]])
        return Dp

    def _makeBp(self):
        # gen Bp
        Bp0 = np.zeros((3,6))
        _,bi,ci,area,_,_,_ = self._makeArea()
        b1,b2,b3,c1,c2,c3 = bi[0],bi[1],bi[2],ci[0],ci[1],ci[2]
        Bp0[0][0],Bp0[0][2],Bp0[0][4] = b1,b2,b3
        Bp0[1][1],Bp0[1][3],Bp0[1][5] = c1,c2,c3
        Bp0[2][0],Bp0[2][1],Bp0[2][2],Bp0[2][3],Bp0[2][4],Bp0[2][5] = c1,b1,c2,b2,c3,b3
        Bp0 = Bp0/(2*area)
        # gen surface area
        self.area = area
        return Bp0

    def _makeKp(self):
        Bp = self._makeBp()
        Dp = self._makeDp()
        Kp = np.dot(np.dot(Bp.T,Dp),Bp) * self.area

        return Kp

    def _makeDb(self):
        Db = (self.young*(self.thickness**3))/(12*((1-(self.poisson**2))))*np.array([[1,self.poisson,0],[self.poisson,1,0],[0,0,(1-self.poisson)/2]])
        return Db

    def _makeBb(self,ip1,ip2,ip3):
        Bbi = np.zeros((3,9))
        L1 = ip1
        L2 = ip2
        L3 = ip3
        ddn0=np.zeros((3,3))
        ddn1=np.zeros((3,3))
        ddn2=np.zeros((3,3))
        ai,bi,ci,area,l,lx,ly = self._makeArea()
        l = [L1,L2,L3,L1,L2]
        a1,a2,a3,b1,b2,b3,c1,c2,c3 = ai[0],ai[1],ai[2],bi[0],bi[1],bi[2],ci[0],ci[1],ci[2]

        lllxx=2.*( lx[0]*lx[1]*l[2] \
              +lx[1]*lx[2]*l[0] \
              +lx[2]*lx[0]*l[1] )
        lllyy=2.*( ly[0]*ly[1]*l[2] \
                  +ly[1]*ly[2]*l[0] \
                  +ly[2]*ly[0]*l[1] )
        lllxy= l[0]*(lx[1]*ly[2]+ly[1]*lx[2])\
              +l[1]*(lx[2]*ly[0]+ly[2]*lx[0])\
              +l[2]*(lx[0]*ly[1]+ly[0]*lx[1])


        for i in range(3):
            j=i+1-3*((i+1)//3)
            k=i+2-3*((i+2)//3)

            ddn0[0,i]=2.*( lx[i]*lx[j]*( l[i]- l[j]) \
                      + l[i]*lx[j]*(lx[i]-lx[j]) \
                      +lx[i]* l[j]*(lx[i]-lx[j]) \
                      +lx[i]*lx[k]*( l[i]- l[k]) \
                      + l[i]*lx[k]*(lx[i]-lx[k]) \
                      +lx[i]* l[k]*(lx[i]-lx[k]) )

            ddn0[1,i]=2.*( ly[i]*ly[j]*( l[i]- l[j]) \
                          + l[i]*ly[j]*(ly[i]-ly[j]) \
                          +ly[i]* l[j]*(ly[i]-ly[j]) \
                          +ly[i]*ly[k]*( l[i]- l[k]) \
                          + l[i]*ly[k]*(ly[i]-ly[k]) \
                          +ly[i]* l[k]*(ly[i]-ly[k]) )

            ddn0[2,i]= (lx[i]*ly[j]+ly[i]*lx[j])*( l[i]- l[j])\
                      +(ly[i]* l[j]+ l[i]*ly[j])*(lx[i]-lx[j])\
                      +(lx[i]* l[j]+ l[i]*lx[j])*(ly[i]-ly[j])\
                      +(lx[i]*ly[k]+ly[i]*lx[k])*( l[i]- l[k])\
                      +(ly[i]* l[k]+ l[i]*ly[k])*(lx[i]-lx[k])\
                      +(lx[i]* l[k]+ l[i]*lx[k])*(ly[i]-ly[k])

            ddn1[0,i]=-bi[k]*(2.*lx[i]*lx[i]*l[j]+4.*l[i]*lx[i]*lx[j]+0.5*lllxx) \
                  +bi[j]*(2.*lx[i]*lx[i]*l[k]+4.*l[i]*lx[i]*lx[k]+0.5*lllxx)
            ddn1[1,i]=-bi[k]*(2.*ly[i]*ly[i]*l[j]+4.*l[i]*ly[i]*ly[j]+0.5*lllyy) \
                      +bi[j]*(2.*ly[i]*ly[i]*l[k]+4.*l[i]*ly[i]*ly[k]+0.5*lllyy)
            ddn1[2,i]=-bi[k]*(2.*ly[i]*lx[i]*l[j]+2.*l[i]*lx[i]*ly[j]\
                                                 +2.*l[i]*ly[i]*lx[j]+0.5*lllxy) \
                      +bi[j]*(2.*ly[i]*lx[i]*l[k]+2.*l[i]*lx[i]*ly[k]\
                                                 +2.*l[i]*ly[i]*lx[k]+0.5*lllxy)


            ddn2[0,i]= ci[k]*(2.*lx[i]*lx[i]*l[j]+4.*l[i]*lx[i]*lx[j]+0.5*lllxx) \
                      -ci[j]*(2.*lx[i]*lx[i]*l[k]+4.*l[i]*lx[i]*lx[k]+0.5*lllxx)
            ddn2[1,i]= ci[k]*(2.*ly[i]*ly[i]*l[j]+4.*l[i]*ly[i]*ly[j]+0.5*lllyy) \
                      -ci[j]*(2.*ly[i]*ly[i]*l[k]+4.*l[i]*ly[i]*ly[k]+0.5*lllyy)
            ddn2[2,i]= ci[k]*(2.*ly[i]*lx[i]*l[j]+2.*l[i]*lx[i]*ly[j]\
                                                 +2.*l[i]*ly[i]*lx[j]+0.5*lllxy) \
                      -ci[j]*(2.*ly[i]*lx[i]*l[k]+2.*l[i]*lx[i]*ly[k]\
                                                 +2.*l[i]*ly[i]*lx[k]+0.5*lllxy)

        for i in range(0,3):
            for j in range(0,3):
                Bbi[i,3*j  ]=-ddn0[i,j]
                Bbi[i,3*j+1]=-ddn1[i,j]
                Bbi[i,3*j+2]=+ddn2[i,j]

        for i in range(0,9):
            Bbi[2,i]=2.*Bbi[2,i]

        return Bbi


    def _makeKb(self):
        Db = self._makeDb()
        Kb = np.zeros((9,9))
        for i in range(len(self.g_weigth)):
            Bb = self._makeBb(self.L1i[i],self.L2i[i],self.L3i[i])
            Kb += np.dot(np.dot(Bb.T,Db),Bb) * self.g_weigth[i] * self.area


        return Kb

    def _makeKt(self):
        Kt = (0.03*self.young*self.thickness*np.array([[1,-0.5,-0.5],[-0.5,1,-0.5],[-0.5,-0.5,1]])*self.area)
        return Kt

    def makeK(self):
        self.K = np.zeros((18,18))
        Kp = self._makeKp()
        Kb = self._makeKb()
        Kt = self._makeKt()

         # Combining stiff. matrices
        for i in range(0,6):
            for j in range(0,6):
                ii=6*(i//2)+(i-2*(i//2))
                jj=6*(j//2)+(j-2*(j//2))
                self.K[ii,jj]=Kp[i,j]

        for i in range(0,9):
            for j in range(0,9):
                ii=6*(i//3)+(i-3*(i//3))+2
                jj=6*(j//3)+(j-3*(j//3))+2
                self.K[ii,jj]=Kb[i,j]

        for i in range(0,3):
            for j in range(0,3):
                ii=6*(i+1)-1
                jj=6*(j+1)-1
                self.K[ii,jj]=Kt[i,j]

class Model():
    def __init__(self):
        self.name = None
        self.loads = []
        self.nodes = []
        self.elements = []
        self.ndof = 0
        self.K_matrix = []
        self.all_force = []
        self.all_u = []
        self.U_full = 0
        self.surface = 0

    def set_name(self,name):
        self.name = name
    def set_loads(self,load):
        self.loads.append(load)

    def set_nodes(self,node):
        self.nodes.append(node)

    def add_elements(self,element):
        self.elements.append(element)

        #element.sort_node()
    def set_elements(self,element):
        element.set_t()
        element.makeK()

    def gen_all(self):
        for i in range(len(self.elements)):
            self.elements[i].set_t()
            self.elements[i].makeK()
        self.set_K()
        self.set_all_force()
        self.cal_all_u()

    def gen_surface(self):
        self.surface = 0
        for i in range(len(self.elements)):
            self.surface+=self.elements[i].area
    def restore(self):
        self.K_matrix = []
        self.all_force = []
        self.all_u = []
        self.U_full = 0
        self.surface = 0
        for i in range(len(self.nodes)):
            self.nodes[i].dof = []
            self.nodes[i].global_d =[[0],[0],[0],[0],[0],[0]]
        for i in range(len(self.elements)):
            self.elements[i].snodes = [] # sorted node
            self.elements[i].element_dof = []
            self.elements[i].lcoord = []
            self.elements[i].area = 0
            #--------------------------------------
            # K of each node
            self.elements[i].K = None
            #--------------------------------------
            # Transformation matirix
            self.elements[i].tmatrixsmall = [] #3x3
            self.elements[i].tmatrix = [] #18x18
            #--------------------------------------
            # obj file
            self.elements[i].obj_node =[]
            self.elements[i].obj_element_start = 0
            self.elements[i].obj_element =[]


    def set_K(self):
        # ---------------------------------
        # [count ndof]
        numdof = 0
        for i in range(len(self.nodes)):
            for j in range(6):
                if self.nodes[i].res[j] == 0:
                    numdof += 1
                    self.nodes[i].dof.append(numdof)
                else:
                    self.nodes[i].dof.append(0)

        self.ndof = numdof
        # [Zero K-matrix]
        self.K_matrix = np.zeros((numdof,numdof))
        #   make ttnsc
        ttnsc = []
        for i in range(len(self.elements)):
            element_ttnsc =[]
            for j in range(len(self.elements[i].nodes)):
                for k in range(6):
                    element_ttnsc.append(self.elements[i].nodes[j].dof[k])
            ttnsc.append(element_ttnsc)


        #   add element_k to K according to num DOF
        for i in range(len(self.elements)):
            # [Transform]
            eleK = np.dot(np.dot(self.elements[i].tmatrix.T,self.elements[i].K),self.elements[i].tmatrix)
            for j in range(18):
                if ttnsc[i][j] > 0:
                    for k in range(18):
                        if ttnsc[i][k] > 0:
                            self.K_matrix[ttnsc[i][j]-1][ttnsc[i][k]-1] += eleK[j][k]

    def set_all_force(self):
        #   Make [0,0,0,0,0,0] force vector from node (6d-array)
        Z_force_3d = []
        for i in range(len(self.nodes)):
            Z_force_3d.append([0,0,0,0,0,0])
        #   Add load to Z_force_3d
        for i in range(len(self.nodes)):
            if len(self.nodes[i].loads) != 0:
                for j in range(len(self.nodes[i].loads)):
                    Z_force_3d[i][0] += self.nodes[i].loads[j].size[0]
                    Z_force_3d[i][1] += self.nodes[i].loads[j].size[1]
                    Z_force_3d[i][2] += self.nodes[i].loads[j].size[2]
                    Z_force_3d[i][3] += self.nodes[i].loads[j].size[3]
                    Z_force_3d[i][4] += self.nodes[i].loads[j].size[4]
                    Z_force_3d[i][5] += self.nodes[i].loads[j].size[5]
        #   Move load from Z_force_3d to self.all_force
        for i in range(self.ndof):
            self.all_force.append([])
        for i in range(len(self.nodes)):
            for j in range(6):
                if self.nodes[i].dof[j] != 0:
                    self.all_force[self.nodes[i].dof[j]-1].append(Z_force_3d[i][j])
                else:
                    pass
    def cal_all_u(self):
        #   calculate U(with DOF)


        u = np.linalg.lstsq(self.K_matrix,self.all_force,rcond=-1)[0]


        #   calculate strain energy
        self.U_full = (np.matmul(u.transpose(),(np.matmul(self.K_matrix,u)))*0.5)[0]
        u.tolist()
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i].dof)):
                if self.nodes[i].dof[j] != 0:
                    self.nodes[i].global_d[j][0] = u[self.nodes[i].dof[j]-1][0]

        #   Put U in all_u[0] according to dof
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i].global_d)):
                self.all_u.append(self.nodes[i].global_d[j])



    def gen_obj(self,name):
        face_count = 0
        for i in range(len(self.elements)):
            self.elements[i].gen_obj_node(face_count)
            face_count += len(self.elements[i].obj_node)
            self.elements[i].gen_obj_element()

        new_file = open(name, "w+")
        # ----------------
        # vertice
        # ----------------
        for i in range(len(self.elements)):
            for j in range(len(self.elements[i].obj_node)):
                new_file.write("v {} {} {}\r\n".format(self.elements[i].obj_node[j][0],self.elements[i].obj_node[j][1],self.elements[i].obj_node[j][2]))
            new_file.write("\n")

        # ----------------
        # faces
        # ----------------

        for i in range(len(self.elements)):
            for j in range(len(self.elements[i].obj_element)):
                if len(self.elements[i].obj_element[j]) == 3:
                    new_file.write("f {} {} {} \r\n".format(self.elements[i].obj_element[j][0],self.elements[i].obj_element[j][1],self.elements[i].obj_element[j][2]))
                    new_file.write("\n")
                elif len(self.elements[i].obj_element[j]) == 4:
                    new_file.write("f {} {} {} {} \r\n".format(self.elements[i].obj_element[j][0],self.elements[i].obj_element[j][1],self.elements[i].obj_element[j][2],self.elements[i].obj_element[j][3]))
                    new_file.write("\n")

        new_file.close()

'''
#--------------------------------------
# TEST
#--------------------------------------
# load
l1 = Load()
l1.set_name(1)
l1.set_size(0,0,-1,0,0,0)

# nodes
n1 = Node()
n1.set_name(1)
n1.set_coord(0,0,0)
n1.set_res(1,1,1,0,0,0)

n2 = Node()
n2.set_name(2)
n2.set_coord(1,0,0)
n2.set_res(1,1,1,0,0,0)

n3 = Node()
n3.set_name(3)
n3.set_coord(2,0,0)
n3.set_res(1,1,1,0,0,0)

n4 = Node()
n4.set_name(4)
n4.set_coord(0,1,0)
n4.set_res(1,1,1,0,0,0)

n5 = Node()
n5.set_name(5)
n5.set_coord(1,1,0.1)
n5.set_res(0,0,0,0,0,0)

n6 = Node()
n6.set_name(6)
n6.set_coord(2,1,0)
n6.set_res(1,1,1,0,0,0)

n7 = Node()
n7.set_name(7)
n7.set_coord(0,2,0)
n7.set_res(1,1,1,0,0,0)

n8 = Node()
n8.set_name(8)
n8.set_coord(1,2,0)
n8.set_res(1,1,1,0,0,0)

n9 = Node()
n9.set_name(9)
n9.set_coord(2,2,0)
n9.set_res(1,1,1,0,0,0)

n5.set_loads(l1)

e1 = Element()
e1.set_name(1)
e1.set_nodes(n5)
e1.set_nodes(n4)
e1.set_nodes(n1)
e1.set_young(0.21e+8)
e1.set_poisson(0.17)
e1.set_thickness(0.2)

e2 = Element()
e2.set_name(2)
e2.set_nodes(n1)
e2.set_nodes(n2)
e2.set_nodes(n5)
e2.set_young(0.21e+8)
e2.set_poisson(0.17)
e2.set_thickness(0.2)

e3 = Element()
e3.set_name(3)
e3.set_nodes(n2)
e3.set_nodes(n3)
e3.set_nodes(n5)
e3.set_young(0.21e+8)
e3.set_poisson(0.17)
e3.set_thickness(0.2)

e4 = Element()
e4.set_name(4)
e4.set_nodes(n6)
e4.set_nodes(n5)
e4.set_nodes(n3)
e4.set_young(0.21e+8)
e4.set_poisson(0.17)
e4.set_thickness(0.2)

e5 = Element()
e5.set_name(5)
e5.set_nodes(n4)
e5.set_nodes(n5)
e5.set_nodes(n7)
e5.set_young(0.21e+8)
e5.set_poisson(0.17)
e5.set_thickness(0.2)

e6 = Element()
e6.set_name(6)
e6.set_nodes(n8)
e6.set_nodes(n7)
e6.set_nodes(n5)
e6.set_young(0.21e+8)
e6.set_poisson(0.17)
e6.set_thickness(0.2)

e7 = Element()
e7.set_name(7)
e7.set_nodes(n5)
e7.set_nodes(n6)
e7.set_nodes(n9)
e7.set_young(0.21e+8)
e7.set_poisson(0.17)
e7.set_thickness(0.2)

e8 = Element()
e8.set_name(8)
e8.set_nodes(n9)
e8.set_nodes(n8)
e8.set_nodes(n5)
e8.set_young(0.21e+8)
e8.set_poisson(0.17)
e8.set_thickness(0.2)



model = Model()
model.name = 1
model.set_nodes(n1)
model.set_nodes(n2)
model.set_nodes(n3)
model.set_nodes(n4)
model.set_nodes(n5)
model.set_nodes(n6)
model.set_nodes(n7)
model.set_nodes(n8)
model.set_nodes(n9)

model.add_elements(e1)
model.add_elements(e2)
model.add_elements(e3)
model.add_elements(e4)
model.add_elements(e5)
model.add_elements(e6)
model.add_elements(e7)
model.add_elements(e8)

for i in range(len(model.elements)):
    model.set_elements(model.elements[i])



model.gen_all()

name = 'Z_test06.obj'
model.gen_obj(name)
'''
