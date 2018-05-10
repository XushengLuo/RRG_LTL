from random import uniform
from networkx import Graph, draw
import numpy as np
import math

class rrg(object):
    """ construction of prefix and suffix tree
    """
    def __init__(self, n_robot, ts, init, step_size):
        """
        :param acpt:  accepting state
        :param ts: transition system
        :param buchi_graph:  Buchi graph
        :param init: product initial state
        """
        self.robot = n_robot
        self.init = init
        self.step_size = step_size
        self.ts = ts
        self.dim = len(self.ts['workspace'])
        self.rrg = Graph(type='wTS', init=init)
        uni_ball = [1, 2, 3.142, 4.189, 4.935, 5.264, 5.168, 4.725, 4.059, 3.299, 2.550]
        self.gamma = np.ceil(4 * np.power(1 / uni_ball[self.robot * self.dim], 1. / (self.dim * self.robot)))  # unit workspace

        label = []
        for i in range(self.robot):
            l = self.label(init[i])
            # exists one sampled point lies within obstacles
            if l != '':
                l = l + str(i+1)
            label.append(l)

        self.rrg.add_node(self.mulp2sglp(init), label=label)

    def sample(self):
        """
        sample point from the workspace
        :return: sampled point, tuple
        """
        x_rand = []
        for i in range(self.dim):
            x_rand.append(uniform(0, self.ts['workspace'][i]))

        return tuple(x_rand)

    def steer(self, x1, x2):
        """
        steer
        :param: x1: master point, center of ball form: single point ()
        :param: x2: form: single point ()
        :return: new point form: single point ()
        """
        if np.linalg.norm(np.subtract(x1, x2)) <= self.step_size:
            return x2
        else:
            return tuple(np.asarray(x1) + self.step_size * (np.subtract(x2, x1))/np.linalg.norm(np.subtract(x2, x1)))

    def far(self, x_r, eta1, eta2):
        """
        find the vertices in the disk between far and near ball, if no vertex in the near ball
        :param x_r: sampled x  form: single point ()
        :param eta1: radius of near ball
        :param eta2: radius of far ball
        :return: vertices in the disk between far and near ball
        """
        x_far = []
        for vertex in self.rrg.nodes:
            dist = np.linalg.norm(np.subtract(x_r, (vertex)))
            if dist <= eta1:
                return []
            elif dist <= eta2:
                x_far.append(vertex)
        return x_far

    def near(self, x_r, eta2):
        """
        find the vertices in the near ball
        :param x_r: sampled x   form: single point
        :param eta2: radius of far ball
        :return: vertices in the near ball form:single point
        """
        x_near = []
        for vertex in self.rrg.nodes:
            if np.linalg.norm(np.subtract(x_r, vertex)) <= eta2:
                x_near.append(vertex)
        return x_near

    def isSimSeg(self, x1, x2, label):
        """
        simple segment
        :param x1: vertex in the ball form:single point
        :param x2: new x form:single point
        :return: dict (x1, x2) true (isSimple)
        """
        return self.obs_check(x1, x2, label)

    def obs_check(self, x1, x2, label):
        """
        check whether obstacle free along the line from x_near to x_new and cross the boundary at most once
        :param x1: vertex in the near ball, form: multiple point      q_near
        :param x2: new state form: multiple point     x_new
        :param label: label of x2
        :return: dict (x1, x2): true (obs_free)
        """

        obs_check_dict = {}
        obs_check_dict[(x1, x2)] = True
        x1m = self.sglp2mulp(x1)
        x2m = self.sglp2mulp(x2)
        flag = True       # indicate whether break and jump to outer loop
        for r in range(self.robot):
            for i in range(1, 11):
                mid = tuple(np.asarray(x1m[r]) + i/10. * np.subtract(x2m[r], x1m[r]))
                mid_label = self.label(mid)
                if mid_label != '':
                    mid_label = mid_label + str(r+1)
                if 'o' in mid_label or (mid_label != self.rrg.nodes[x1]['label'][r] and mid_label != label[r]):
                    # obstacle             pass through one region more than once
                    obs_check_dict[(x1, x2)] = False
                    flag = False
                    break
            if not flag:
                break

        return obs_check_dict

    def label(self, x):
        """
        generating the label of position state
        :param x: position
        :return: label
        """
        # whether x lies within obstacle
        for (obs, boundary) in iter(self.ts['obs'].items()):
            if obs[1] == 'b' and np.linalg.norm(np.subtract(x, boundary[0:-1])) <= boundary[-1]:
                return obs[0]
            elif obs[1] == 'p':
                dictator = True
                for i in range(len(boundary)):
                    if np.dot(x, boundary[i][0:-1]) + boundary[i][-1] > 0:
                        dictator = False
                        break
                if dictator == True:
                    return obs[0]


        # whether x lies within regions
        for (regions, boundary) in iter(self.ts['region'].items()):
            if regions[1] == 'b' and np.linalg.norm(x - np.asarray(boundary[0:-1])) <= boundary[-1]:
                return regions[0]
            elif regions[1] == 'p':
                dictator = True
                for i in range(len(boundary)):
                    if np.dot(x, np.asarray(boundary[i][0:-1])) + boundary[i][-1] > 0:
                        dictator = False
                        break
                if dictator == True:
                    return regions[0]

        return ''

    def checkTranB(self, b_state, x_label, q_b_new):
        """ decide valid transition, whether b_state --L(x)---> q_b_new
             Algorithm2 in Chapter 2 Motion and Task Planning
             :param b_state: buchi state
             :param x_label: label of x
             :param q_b_new buchi state
             :return True satisfied
        """
        b_state_succ = self.buchi_graph.succ[b_state]
        # q_b_new is not the successor of b_state
        if q_b_new not in b_state_succ:
             return False

        b_label = self.buchi_graph.edges[(b_state, q_b_new)]['label']
        if self.t_satisfy_b(x_label, b_label):
            return True

    def t_satisfy_b(self, x_label, b_label):
        """ decide whether label of self.ts_graph can satisfy label of self.buchi_graph
            :param x_label: label of x
            :param b_label: label of buchi state
            :return t_s_b: true if satisfied
        """
        t_s_b = True
        # split label with ||
        b_label = b_label.split('||')
        for label in b_label:
            t_s_b = True
            # spit label with &&
            atomic_label = label.split('&&')
            for a in atomic_label:
                a = a.strip()
                if a == '1':
                    continue
                # whether ! in an atomic proposition
                if '!' in a:
                    if a[1:] in x_label:
                        t_s_b = False
                        break
                else:
                    if not a in x_label:
                       t_s_b = False
                       break
            # either one of || holds
            if t_s_b:
                return t_s_b
        return t_s_b

    def mulp2sglp(self, point):
        """
        convert multiple form point ((),(),(),...) to single form point ()
        :param point: multiple points ((),(),(),...)
        :return: signle point ()
        """
        sp = []
        for p in point:
            sp = sp + list(p)
        return tuple(sp)

    def sglp2mulp(self, point):
        """
        convert single form point () to multiple form point ((), (), (), ...)
        :param point: single form point ()
        :return:  multiple form point ((), (), (), ...)
        """
        mp = []
        for i in range(self.robot):
            mp.append(point[i*self.dim :(i+1)*self.dim])
        return tuple(mp)

