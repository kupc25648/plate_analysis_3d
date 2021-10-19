'''
======================================================
FINITE ELEMENT PROGRAM FOR MECHANIC ANALSIS
MODEL : 3 nodes plate element
BY : CHI-TATHON KUPWIWAT
TEST MODEL

This file create structural model file from various types of input

Input :
        1. Hard code
        2. Generate algorithm
        3. Read from .txt / .csv
        4. Read from other file format

Output : structural model class

Using   : structural_analysis.py
          structural_optimization.py (not required)
Used by : structural_game.py

Update : 2020 01


======================================================
'''

from FEM_triplate import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random


class gen_model:
    def __init__(self,
                num_x,
                num_y,
                lx,
                ly,
                loadx,
                loady,
                loadz,
                Young,
                Poisson,
                Thickness,
                c1,c2,c3,c4,c5,c6,c7,
                forgame=None,game_max_z_val=None):
        # Load
        self.load = Load()
        self.load.set_name(1)
        self.load.set_size(loadx,loady,loadz,0,0,0)
        # Plate Dimensions
        if (num_x%2!=0) or (num_x%2!=0):
            print('WARNING: num_x and num_y must be even numbers')
        self.num_x = num_x #oddnumber only
        self.num_y = num_y #oddnumber only
        self.lx = lx
        self.ly = ly
        self.lz = 0
        self.y_column = []
        self._gen_y_column()
        # Shell Properties
        self.Young = Young
        self.Poisson = Poisson
        self.Thickness = Thickness
        # Generate lists and numbers
        self.alln = [] # Nodes list
        self.alle = [] # Element list
        self.num_node = 1 # For tracking and naming nodesself.
        self.Allex = [] # list for elements in initial X direction
        self.Alle0 = [] # list for elements in initial Y direction
        self.Alley = [] # list for elements in all Y direction
        #   Inititial Point
        self.Xi = -(self.num_x*self.lx)/2
        self.Yi = -(self.num_y*self.ly)/2
        self.Zi = 0
        # boundary condition
        self.minx = self.Xi
        self.maxx = self.Xi + self.num_x * self.lx
        self.miny = self.Yi
        self.maxy = self.Yi + self.num_y * self.ly
        self.centerx = (self.maxx+self.minx)/2
        self.cemtery = (self.maxy+self.miny)/2
        # structural data[y][x]
        self.n_u_name_div =[]
        for i in range(self.num_y+1):
            self.n_u_name_div.append([])
            for j in range(self.num_x+1):
                self.n_u_name_div[-1].append(None)

        # structural generation/optimization
        self.rand_bd=None
        rand_val = random.random()
        if rand_val >= 0.5:
            self.rand_bd = 0 # Dome
        else:
            self.rand_bd = 1 # Isler


        # -------------------------------------------------
        # Parameter function
        # Z = [ c1*(x**2) + c2*(x*y) + c3*(y**2) + c4*x + c5*y + c6 ] * c7
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.c7 = c7
        # -------------------------------------------------
        # Reinforcement Learning Defaults / 強化学習のデフォルト
        # 0: minimal surface game / 最小限の表面ゲーム
        # 1: structural generation game / 構造生成ゲーム
        # 2: structural optimization game / 構造最適化ゲーム
        self.forgame = forgame
        # Maximum y coordinate(m.) / 最大y座標（m。）
        self.game_max_z_val = game_max_z_val
        # -------------------------------------------------
        # Frame Surface (will be generated after init) / フレームモデル（初期化後に生成されます）
        self.surface_1 = 0
        # -------------------------------------------------
        # Shell Model
        self.model = None # will be generated after init
        self.generate()
    def gen_surface1(self):
        self.model.gen_surface()
        self.surface_1 = [self.model.surface]

    def _gen_y_column(self):
        for i in range(self.num_x):
            self.y_column.append([])

    # PARAMETRIC FUNCTION
    def set_Z(self,listxy):
        #self.c6 = random.random()*10 # for hack randomness
        Z =((self.c1*(listxy[0]**2))+(self.c2*(listxy[0]*listxy[1]))+(self.c3*(listxy[1]**2))+(self.c4*(listxy[0]))+(self.c5*(listxy[1]))+self.c6)*self.c7

        # parabolic shell
        #Z = ((listxy[0]**2)+(listxy[1]**2))*(-0.2)
        # flate plate
        #Z = 0
        # Test sin function
        #Z = np.sin(listxy[0]) + np.sin(listxy[1])
        # parabolic shell with randomness
        #Z = ((listxy[0]**2)+(listxy[1]**2) + (round(np.random.random(),2)*5))*(-0.3)
        #Z = -0.05*((listxy[0]**2) + (listxy[1]**2)) + np.sin((listxy[0]**2) )
        Z = listxy[0]*(listxy[0]**2-(3*(listxy[1]**2)))*0.05 #monkey sanddle
        return Z


    def Gen_node(self,px,py,pz,rx,ry,rz,rtx,rty,rtz,num_node):
        x = 'n'+str(num_node)
        x = Node()
        x.set_name(num_node)
        x.set_coord(px,py,pz)
        x.set_res(rx,ry,rz,rtx,rty,rtz)
        #----------------------------------------------------------
        #   SET RESTRICTION SURROUND NODES
        #   SET HEIGHT
        #   Add rule to set initial node height here
        #----------------------------------------------------------
        '''
        ---------------------------------------------
        Generate Initial shape for reinforcement learning / 強化学習の初期形状を生成する
        ---------------------------------------------
        '''
        if self.forgame == 0000:
            #print('-------------------------------------------------------')
            #print('MINIMAL SURFACE GAME') #研究室
            #print('-------------------------------------------------------')
            # Random initial node height / ランダムな初期ノードの高さ
            x.coord[2] = random.randint(0,int(self.game_max_z_val*10))/int(self.game_max_z_val*10)
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) / 周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            if (px == self.minx):
                x.set_res(1,1,1,1,1,1)
                x.coord[2] = (x.coord[1]-self.miny)*self.game_max_z_val/(self.maxy-self.miny)
            else:
                pass
            if (py == self.miny):
                x.set_res(1,1,1,1,1,1)
                x.coord[2] = (x.coord[0]-self.minx)*self.game_max_z_val/(self.maxx-self.minx)
            else:
                pass
            if (px == self.maxx):
                x.set_res(1,1,1,1,1,1)
                x.coord[2] = self.game_max_z_val-((x.coord[1]-self.miny)*self.game_max_z_val/(self.maxy-self.miny))
            else:
                pass
            if (py == self.maxy):
                x.set_res(1,1,1,1,1,1)
                x.coord[2] = self.game_max_z_val-((x.coord[0]-self.minx)*self.game_max_z_val/(self.maxx-self.minx))
            else:
                pass
        elif self.forgame ==1000:
            #print('-------------------------------------------------------')
            #print('GENERATE GAME') #研究室
            #print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            x.coord[2] = 0
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            if self.rand_bd == 0:
                if (px == self.minx) or (py == self.miny) or (px == self.maxx) or (py == self.maxy):
                    x.set_res(1,1,1,1,1,1)
                else:
                    pass
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            else:
                if ((px == self.minx) and (py == self.miny)) or ((px == self.maxx) and (py == self.maxy)) or ((px == self.maxx) and (py == self.miny)) or ((px == self.minx) and (py == self.maxy)):
                    x.set_res(1,1,1,1,1,1)
                else:
                    pass
        elif self.forgame ==1001:
            #print('-------------------------------------------------------')
            #print('GENERATE GAME') #研究室
            #print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            x.coord[2] = 0
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            if (px == self.minx) or (py == self.miny) or (px == self.maxx) or (py == self.maxy):
                x.set_res(1,1,1,1,1,1)
            else:
                pass
        elif self.forgame ==1002:
            #print('-------------------------------------------------------')
            #print('GENERATE GAME') #研究室
            #print('-------------------------------------------------------')
            # Initial node height equal to 0 / 最初のノードの高さは0です
            x.coord[2] = 0
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            if ((px == self.minx) and (py == self.miny)) or ((px == self.maxx) and (py == self.maxy)) or ((px == self.maxx) and (py == self.miny)) or ((px == self.minx) and (py == self.maxy)):
                x.set_res(1,1,1,1,1,1)
            else:
                pass

        elif self.forgame ==2000:
            #print('-------------------------------------------------------')
            #print('OPTIMIZATION GAME') #研究室
            #print('-------------------------------------------------------')
            # Initial node height equal to random / 最初のノードの高さは0です
            x.coord[2] = (random.randint(0,self.game_max_z_val))/self.game_max_z_val
            rand_val = random.random()
            # Surrounding nodes set_res to (1,1,1,1,1,1) (fixed joint) /周囲のノードset_resを（1,1,1,1,1,1）に（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            if self.rand_bd == 0:
                if (px == self.minx) or (py == self.miny) or (px == self.maxx) or (py == self.maxy):
                    x.set_res(1,1,1,1,1,1)
                else:
                    pass
            # Corner nodes set_res to (1,1,1,1,1,1) (fixed joint) / コーナーノードset_resを（1,1,1,1,1,1）（固定ジョイント）
            # Set Z axis to 0 / Z軸を0に設定
            else:
                if ((px == self.minx) and (py == self.miny)) or ((px == self.maxx) and (py == self.maxy)) or ((px == self.maxx) and (py == self.miny)) or ((px == self.minx) and (py == self.maxy)):
                    x.set_res(1,1,1,1,1,1)
                else:
                    pass



        else: # defult
            x.coord[2] = self.set_Z(x.coord) - self.lz/2 # set height
            if (px == self.minx ) or (px == self.maxx ):
                x.set_res(1,1,1,0,0,0)
            elif (py == self.miny ) or (py == self.maxy ):
                x.set_res(1,1,1,0,0,0)
            else:
                x.set_res(0,0,0,0,0,0)

        #----------------------------------------------------------
        #   SET LOAD TO UPPER NODES
        #----------------------------------------------------------
        '''
        maxz = self.Zi + lz/2
        if pz == maxz :
            x.set_loads(l1)
        '''

        #if (round(px,2) == round(centerx,2) ) and (round(py,2) == round(cemtery,2) ):
        x.set_loads(self.load)

        self.alln.append(x)

    def savetxt(self,name):
        # ------------------------------
        # Write and save output model  / 出力モデルファイルの書き込みと保存
        # ------------------------------
        new_file = open(name, "w+")
        for num1 in range(len(self.model.loads)):
            new_file.write("{}\r\n".format(self.model.loads[num1]))
        for num1 in range(len(self.model.nodes)):
            new_file.write("{}\r\n".format(self.model.nodes[num1]))
        for num1 in range(len(self.model.elements)):
            new_file.write("{}\r\n".format(self.model.elements[num1]))
        new_file.close()

    def generate(self):

        #----------------------------------------------------------
        #   Algorithm 1
        #       Create 4 initial nodes
        #       Create 2 initial elements
        #----------------------------------------------------------
        # COORDINATE LIST FOR INIIAL NODES
        dx = (self.lx)
        dy = (self.ly)
        N01 = [ self.Xi + 0  , self.Yi + 0  , self.Zi ]
        N02 = [ self.Xi + dx , self.Yi + 0  , self.Zi ]
        N03 = [ self.Xi + 0  , self.Yi + dy , self.Zi ]
        N04 = [ self.Xi + dx , self.Yi + dy , self.Zi ]
        Nlist = [N01,N02,N03,N04]
        for i in range(len(Nlist)):
            self.Gen_node(Nlist[i][0],Nlist[i][1],Nlist[i][2],0,0,0,0,0,0,self.num_node)
            self.num_node += 1

        # first element
        e1 = Element()
        e1.set_name(1)
        e1.set_young(self.Young)
        e1.set_poisson(self.Poisson)
        e1.set_thickness(self.Thickness)
        # set node in counter clockwise direction
        e1.set_nodes(self.alln[0]) # node 1
        e1.set_nodes(self.alln[1]) # node 2
        e1.set_nodes(self.alln[2]) # node 3

        # second element
        e2 = Element()
        e2.set_name(2)
        e2.set_young(self.Young)
        e2.set_poisson(self.Poisson)
        e2.set_thickness(self.Thickness)
        # set node in counter  clockwise direction
        e2.set_nodes(self.alln[2]) # node 3
        e2.set_nodes(self.alln[1]) # node 2
        e2.set_nodes(self.alln[3]) # node 4

        # Put element in element_list and y_column
        self.alle.append(e1)
        self.alle.append(e2)
        self.y_column[0].append([e1,e2]) # put in y_column as a pair


        #----------------------------------------------------------
        #   Algorithm 2
        #       Create 2 nodes
        #       Create elements in initial X direction
        #       2 sub algorithm
        #----------------------------------------------------------

        self.Allex = [[e1,e2]] # list for elements in initial X direction as a pair

        for i in range(1, self.num_x):
            if (i==1) or (i%2 != 0):
                # COORDINATE LIST FOR INIIAL NODES
                dx = (self.lx)
                dy = (self.ly)
                N01 = self.Allex[-1][1].nodes[1]

                N02 =[]
                for num in range(3):
                    N02.append(self.Allex[-1][1].nodes[1].coord[num])
                N02[0] += dx

                N03 = self.Allex[-1][1].nodes[2]

                N04 =[]
                for num in range(3):
                    N04.append(self.Allex[-1][1].nodes[2].coord[num])
                N04[0] += dx

                Nlist = [N01,N02,N03,N04]
                # first element
                y1 = 'e'+str(len(self.alle)+1)
                y1 = Element()
                y1.set_name(len(self.alle)+1)
                y1.set_young(self.Young)
                y1.set_poisson(self.Poisson)
                y1.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y1.set_nodes(Nlist[2]) # node 3
                y1.set_nodes(Nlist[0]) # node 1
                self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4-gen
                self.num_node += 1
                y1.set_nodes(self.alln[-1])

                self.alle.append(y1)
                # second element
                y2 = 'e'+str(len(self.alle)+1)
                y2 = Element()
                y2.set_name(len(self.alle)+1)
                y2.set_young(self.Young)
                y2.set_poisson(self.Poisson)
                y2.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y2.set_nodes(self.alln[-1]) # node 4
                y2.set_nodes(Nlist[0]) # node 1
                self.Gen_node(Nlist[1][0],Nlist[1][1],Nlist[1][2],0,0,0,0,0,0,self.num_node) # node 2-gen
                self.num_node += 1
                y2.set_nodes(self.alln[-1]) # node 2



                self.alle.append(y2)

                self.Allex.append([y1,y2])
                self.y_column[i].append([y1,y2]) # put in y_column as a pair
            else:
                # COORDINATE LIST FOR INIIAL NODES
                dx = (self.lx)
                dy = (self.ly)
                N01 = self.Allex[-1][1].nodes[2]

                N02 =[]
                for num in range(3):
                    N02.append(self.Allex[-1][1].nodes[2].coord[num])
                N02[0] += dx

                N03 = self.Allex[-1][1].nodes[0]

                N04 =[]
                for num in range(3):
                    N04.append(self.Allex[-1][1].nodes[0].coord[num])
                N04[0] += dx

                Nlist = [N01,N02,N03,N04]
                # first element
                y1 = 'e'+str(len(self.alle)+1)
                y1 = Element()
                y1.set_name(len(self.alle)+1)
                y1.set_young(self.Young)
                y1.set_poisson(self.Poisson)
                y1.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y1.set_nodes(Nlist[0]) # node 1
                self.Gen_node(Nlist[1][0],Nlist[1][1],Nlist[1][2],0,0,0,0,0,0,self.num_node) # node 2 -gen
                self.num_node += 1
                y1.set_nodes(self.alln[-1]) # node 2
                y1.set_nodes(Nlist[2]) # node 3

                self.alle.append(y1)
                # second element
                y2 = 'e'+str(len(self.alle)+1)
                y2 = Element()
                y2.set_name(len(self.alle)+1)
                y2.set_young(self.Young)
                y2.set_poisson(self.Poisson)
                y2.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y2.set_nodes(Nlist[2]) # node 3
                y2.set_nodes(self.alln[-1]) # node 2
                self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4 -gen
                self.num_node += 1
                y2.set_nodes(self.alln[-1]) # node 4

                self.alle.append(y2)



                self.Allex.append([y1,y2])
                self.y_column[i].append([y1,y2]) # put in y_column as a pair


        #----------------------------------------------------------
        #   Algorithm 3
        #       Create 18 nodes
        #       Create elements in initial Y direction
        #----------------------------------------------------------

        self.Alle0 = [[e1,e2]] # list for elements in initial Y direction as a pair

        for i in range(1, self.num_y):

            if (i%2!=0):

                # COORDINATE LIST FOR INIIAL NODES
                dx = (self.lx)
                dy = (self.ly)
                N01 = self.Alle0[-1][1].nodes[0]
                N02 = self.Alle0[-1][1].nodes[2]

                N03 =[]
                for num in range(3):
                    N03.append(self.Alle0[-1][1].nodes[0].coord[num])
                N03[1] += dy

                N04 =[]
                for num in range(3):
                    N04.append(self.Alle0[-1][1].nodes[2].coord[num])
                N04[1] += dy

                Nlist = [N01,N02,N03,N04]
                # first element
                y1 = 'e'+str(len(self.alle)+1)
                y1 = Element()
                y1.set_name(len(self.alle)+1)
                y1.set_young(self.Young)
                y1.set_poisson(self.Poisson)
                y1.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                self.Gen_node(Nlist[2][0],Nlist[2][1],Nlist[2][2],0,0,0,0,0,0,self.num_node) # node 3 -gen
                self.num_node += 1
                y1.set_nodes(self.alln[-1])
                y1.set_nodes(Nlist[0]) # node 1
                self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4 -gen
                self.num_node += 1
                y1.set_nodes(self.alln[-1])

                self.alle.append(y1)
                # second element
                y2 = 'e'+str(len(self.alle)+1)
                y2 = Element()
                y2.set_name(len(self.alle)+1)
                y2.set_young(self.Young)
                y2.set_poisson(self.Poisson)
                y2.set_thickness(self.Thickness)
                # set node in clockwise direction
                y2.set_nodes(self.alln[-1]) # node 4
                y2.set_nodes(Nlist[0]) # node 1
                y2.set_nodes(Nlist[1]) # node 2

                self.alle.append(y2)



                self.Alle0.append([y1,y2])
                self.y_column[0].append([y1,y2]) # put in y_column as a pair
            else:
                # COORDINATE LIST FOR INIIAL NODES
                dx = (self.lx)
                dy = (self.ly)
                N01 = self.Alle0[-1][0].nodes[0]
                N02 = self.Alle0[-1][0].nodes[2]

                N03 =[]
                for num in range(3):
                    N03.append(self.Alle0[-1][0].nodes[0].coord[num])
                N03[1] += dy

                N04 =[]
                for num in range(3):
                    N04.append(self.Alle0[-1][0].nodes[2].coord[num])
                N04[1] += dy

                Nlist = [N01,N02,N03,N04]
                # first element
                y1 = 'e'+str(len(self.alle)+1)
                y1 = Element()
                y1.set_name(len(self.alle)+1)
                y1.set_young(self.Young)
                y1.set_poisson(self.Poisson)
                y1.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y1.set_nodes(Nlist[0]) # node 1
                y1.set_nodes(Nlist[1]) # node 2
                self.Gen_node(Nlist[2][0],Nlist[2][1],Nlist[2][2],0,0,0,0,0,0,self.num_node) # node 3 -gen
                self.num_node += 1
                y1.set_nodes(self.alln[-1])

                self.alle.append(y1)
                # second element
                y2 = 'e'+str(len(self.alle)+1)
                y2 = Element()
                y2.set_name(len(self.alle)+1)
                y2.set_young(self.Young)
                y2.set_poisson(self.Poisson)
                y2.set_thickness(self.Thickness)
                # set node in counter clockwise direction
                y2.set_nodes(self.alln[-1]) # node 3
                y2.set_nodes(Nlist[1]) # node 2
                self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4 -gen
                self.num_node += 1
                y2.set_nodes(self.alln[-1])

                self.alle.append(y2)

                self.Alle0.append([y1,y2])
                self.y_column[0].append([y1,y2]) # put in y_column as a pair


        #----------------------------------------------------------
        #   Algorithm 4
        #       Create 1 node
        #       Create elements in all Y direction
        #----------------------------------------------------------

        self.Alley = []  # list for elements in all Y direction
        for i in range(len(self.Alle0)):
            self.Alley.append([self.Alle0[i]])
        self.Alley[0] = self.Allex

        for i in range(1,self.num_x):
            for j in range(1, self.num_y):
                if (i%2==0 and j%2==0) or (i%2!=0 and j%2!=0):
                    # COORDINATE LIST FOR INIIAL NODES
                    dx = (self.lx)
                    dy = (self.ly)
                    N01 = self.Alley[j-1][i][0].nodes[0]
                    N02 = self.Alley[j-1][i][0].nodes[2]
                    N03 = self.Alley[j][-1][1].nodes[0]

                    N04 =[]
                    for num in range(3):
                        N04.append(self.Alley[j][-1][1].nodes[0].coord[num])
                    N04[0] += dx

                    Nlist = [N01,N02,N03,N04]
                    # first element
                    y1 = 'e'+str(len(self.alle)+1)
                    y1 = Element()
                    y1.set_name(len(self.alle)+1)
                    y1.set_young(self.Young)
                    y1.set_poisson(self.Poisson)
                    y1.set_thickness(self.Thickness)
                    # set node in counter clockwise direction
                    y1.set_nodes(Nlist[0]) # node 1
                    y1.set_nodes(Nlist[1]) # node 2
                    y1.set_nodes(Nlist[2]) # node 3
                    self.alle.append(y1)
                    # second element
                    y2 = 'e'+str(len(self.alle)+1)
                    y2 = Element()
                    y2.set_name(len(self.alle)+1)
                    y2.set_young(self.Young)
                    y2.set_poisson(self.Poisson)
                    y2.set_thickness(self.Thickness)
                    # set node in counter clockwise direction
                    y2.set_nodes(Nlist[2]) # node 3
                    y2.set_nodes(Nlist[1]) # node 2
                    self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4 -gen
                    self.num_node += 1
                    y2.set_nodes(self.alln[-1])

                    self.alle.append(y2)

                    self.Alley[j].append([y1,y2])
                    self.y_column[i].append([y1,y2])
                elif (i%2!=0 and j%2==0) or (i%2==0 and j%2!=0):
                    # COORDINATE LIST FOR INIIAL NODES
                    dx = (self.lx)
                    dy = (self.ly)
                    N01 = self.Alley[j-1][i][1].nodes[0]
                    N02 = self.Alley[j-1][i][1].nodes[2]
                    N03 = self.Alley[j][-1][1].nodes[2]

                    N04 =[]
                    for num in range(3):
                        N04.append(self.Alley[j][-1][1].nodes[2].coord[num])
                    N04[0] += dx

                    Nlist = [N01,N02,N03,N04]
                    # first element
                    y1 = 'e'+str(len(self.alle)+1)
                    y1 = Element()
                    y1.set_name(len(self.alle)+1)
                    y1.set_young(self.Young)
                    y1.set_poisson(self.Poisson)
                    y1.set_thickness(self.Thickness)
                    # set node in counter clockwise direction
                    y1.set_nodes(Nlist[2]) # node 3
                    y1.set_nodes(Nlist[0]) # node 1
                    self.Gen_node(Nlist[3][0],Nlist[3][1],Nlist[3][2],0,0,0,0,0,0,self.num_node) # node 4 -gen
                    self.num_node += 1
                    y1.set_nodes(self.alln[-1])

                    self.alle.append(y1)
                    # second element
                    y2 = 'e'+str(len(self.alle)+1)
                    y2 = Element()
                    y2.set_name(len(self.alle)+1)
                    y2.set_young(self.Young)
                    y2.set_poisson(self.Poisson)
                    y2.set_thickness(self.Thickness)
                    # set node in counter clockwise direction
                    y2.set_nodes(self.alln[-1])
                    y2.set_nodes(Nlist[0]) # node 1
                    y2.set_nodes(Nlist[1]) # node 2

                    self.alle.append(y2)

                    self.Alley[j].append([y1,y2])
                    self.y_column[i].append([y1,y2])


        #----------------------------------------------------------
        #   GENERATE MODEL
        #----------------------------------------------------------
        self.model = Model()
        self.model.set_name(1)
        self.model.set_loads(self.load)
        for i in range(len(self.alln)):
            self.model.set_nodes(self.alln[i])
        for i in range(len(self.alle)):
            self.model.add_elements(self.alle[i])
        '''
        #----------------------------------------------------------
        #       Adjust z-coord of all nodes
        #----------------------------------------------------------
        for i in range(len(self.model.elements)):
            for j in range(len(self.model.elements[i].nodes)):
                self.model.elements[i].nodes[j].coord[2] = self.set_Z(self.model.elements[i].nodes[j].coord) - self.lz/2
        '''
        # Setup self.n_u_name_div
        xcoord = []
        ycoord = []
        for i in range(len(self.model.nodes)):
            xcoord.append(self.model.nodes[i].coord[0])
            ycoord.append(self.model.nodes[i].coord[1])
        xcoord = list(dict.fromkeys(xcoord))
        ycoord = list(dict.fromkeys(ycoord))

        for i in range(len(ycoord)):
            for j in range(len(xcoord)):
                for k in range(len(self.model.nodes)):
                    if (self.model.nodes[k].coord[0]==xcoord[j]) and (self.model.nodes[k].coord[1]==ycoord[i]):
                        self.n_u_name_div[i][j]=self.model.nodes[k]



        for i in range(len(self.model.elements)):
            self.model.set_elements(self.model.elements[i])

        self.model.gen_all()

    def render(self,name='',iteration=0,initenergy=0,recentenergy=0):
        fig = plt.figure(figsize=plt.figaspect(0.3))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.75, wspace=0, hspace=None)
        num_element = str(len(model_X.model.elements))

        ax1 = fig.add_subplot(1,4,1,projection='3d')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')
        ax1.view_init(azim=45, elev=45)
        Analysis_type0 = 'RL'
        Structure_type0 = 'Structure type : plate3nodes'
        Model_code0 = 'Size : ' + str(len(self.model.nodes))+' nodes ' + str(len(self.model.elements))+' members'
        Node_and_Member = 'Perspective'
        plt.title(Analysis_type0+'\n'+Structure_type0+'\n'+Model_code0+'\n'+'\n'+Node_and_Member,loc='left')
        plt.grid()
        plt.axis('equal')


        ax2 = fig.add_subplot(1,4,3,projection='3d')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')
        ax2.view_init(azim=270, elev=0)
        Analysis_type0 = ''
        Structure_type0 = ''
        Model_code0 = ''
        Node_and_Member = 'Elevation-1'
        plt.title(Analysis_type0+'\n'+Structure_type0+'\n'+Model_code0+'\n'+'\n'+Node_and_Member,loc='left')
        plt.grid()
        plt.axis('equal')


        ax3 = fig.add_subplot(1,4,4,projection='3d')
        ax3.set_xlabel('X axis')
        ax3.set_ylabel('Y axis')
        ax3.set_zlabel('Z axis')
        ax3.view_init(azim=180, elev=0)
        Analysis_type2 = 'Iteration: '+str(iteration)
        Structure_type2 = 'Base Strain Energy     : ' + str(round(initenergy,9))
        Model_code2 =     'Current Strain Energy : ' + str(round(recentenergy,9))
        plt.title(Analysis_type2+'\n'+Structure_type2+'\n'+Model_code2+'\n'+'\n'+'Elevation-2',loc='left')
        plt.grid()
        plt.axis('equal')


        #plan
        axplan = fig.add_subplot(1,4,2,projection='3d')
        axplan.set_xlabel('X axis')
        axplan.set_ylabel('Y axis')
        axplan.set_zlabel('Z axis')
        axplan.view_init(azim=0, elev=90)
        Analysis_type2 = ''
        Structure_type2 = ''
        Model_code2 =     ''
        plt.title(Analysis_type2+'\n'+Structure_type2+'\n'+Model_code2+'\n'+'\n'+'Plan',loc='left')
        plt.grid()
        plt.axis('equal')
        #ax.set_zlim3d(-2,2)

        #Data
        for i in range(len(self.model.elements)):
            def FEM3Dtrisurfall():
                x = []
                y = []
                z = []

                resx = []
                resy = []
                resz = []

                loadx = []
                loady = []
                loadz = []

                for i in range(len(self.model.elements)):

                    for j in range(len(self.model.elements[i].nodes)):
                        x.append(self.model.elements[i].nodes[j].coord[0]+self.model.elements[i].nodes[j].global_d[0][0])
                        y.append(self.model.elements[i].nodes[j].coord[1]+self.model.elements[i].nodes[j].global_d[1][0])
                        z.append(self.model.elements[i].nodes[j].coord[2]+self.model.elements[i].nodes[j].global_d[2][0])



                for i in range(len(self.model.nodes)):
                    '''
                    SUPPORT
                    '''
                    if (self.model.nodes[i].res[2]) != 0:
                        resx.append(float(self.model.nodes[i].coord[0]))
                        resy.append(float(self.model.nodes[i].coord[1]))
                        resz.append(float(self.model.nodes[i].coord[2]))
                    '''
                    LOAD
                    '''
                    if len(self.model.nodes[i].loads) != 0:
                        loadx.append(float(self.model.nodes[i].coord[0]))
                        loady.append(float(self.model.nodes[i].coord[1]))
                        loadz.append(float(self.model.nodes[i].coord[2]))


                x = np.array(x)
                y = np.array(y)
                z = np.array(z)

                resx = np.array(resx)
                resy = np.array(resy)
                resz = np.array(resz)

                loadx = np.array(loadx)
                loady = np.array(loady)
                loadz = np.array(loadz)

                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
                # Comment or uncomment following both lines to test the fake bounding box:

                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax1.plot([xb], [yb], [zb], 'w')

                ax1.plot_trisurf(x, y, z, color='red', linewidth=0.2, antialiased=True)

                ax1.scatter(resx, resy, resz, color='black', marker = '^')

                #ax1.scatter(loadx, loady, loadz, color='blue',s=10, alpha = 0.5)


                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax2.plot([xb], [yb], [zb], 'w')

                ax2.plot_trisurf(x, y, z, color='red', linewidth=0.2, antialiased=True)

                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   ax3.plot([xb], [yb], [zb], 'w')

                ax3.plot_trisurf(x, y, z, color='red', linewidth=0.2, antialiased=True)

                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
                Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(x.max()+x.min())
                Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(y.max()+y.min())
                Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(z.max()+z.min())
                # Comment or uncomment following both lines to test the fake bounding box:
                for xb, yb, zb in zip(Xb, Yb, Zb):
                   axplan.plot([xb], [yb], [zb], 'w')

                axplan.plot_trisurf(x, y, z, color='red', linewidth=0.2, antialiased=True)


            FEM3Dtrisurfall()
        #plt.savefig(name)
        #plt.close("all")
        plt.show()

# TEST
#-----------------------------
#Params
num_x = 6
num_y = 6

lx  = 1
ly  = 1
loadx = 0
loady = 0
loadz = -1000

Young = 17*1000000000
Poisson = 0.2
Thickness = 0.01

c1 = 1
c2 = 0
c3 = 1
c4 = 0
c5 = 0
c6 = 0
c7 = 0
forgame=None
game_max_z_val=1

#-----------------------------

model_X = gen_model(num_x,num_y,lx,ly,loadx,loady,loadz,Young,Poisson,Thickness,c1,c2,c3,c4,c5,c6,c7,forgame,game_max_z_val)

# to set new function see PARAMETRIC FUNCTION def set_Z(self,listxy): in line 137

name = 'FEM_testsamll.obj'
model_X.model.gen_obj(name) # save .obj file

#print(model_X.model.U_full) # strain energy (Nm.)
#model_X.model.gen_surface()
#print(model_X.model.surface)# surface area (sq.m.)


