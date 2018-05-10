from Problem import problemFormulation
from RRG4MulR import rrg
import datetime
from Buchi import buchi_graph
import networkx as nx
import matplotlib.pyplot as plt
import math
from ConsP import P
import numpy as np
from WorkspacePlot import path_plot


def construction_rrg(rrg, P):
    k = 0
    while True:
        k = k +1
        print(k)
        # initialization
        Xp = []
        Dp = []
        final1 = []
        final2 = []
        eta1 = 1 / math.sqrt(math.pi) * math.pow(math.gamma(rrg.dim * rrg.robot / 2 + 1) / rrg.rrg.number_of_nodes(),
                                                 1 / (rrg.dim * rrg.robot))
        eta2 = 2 * eta1
        # eta2 = rrg.gamma * np.power(np.log(rrg.rrg.number_of_nodes()+1)/rrg.rrg.number_of_nodes(),1./(rrg.dim*rrg.robot))
        # sample form: multiple
        x_rand = list()
        for i in range(rrg.robot):
            x_rand.append(rrg.sample())
        x_rand = rrg.mulp2sglp(tuple(x_rand))

        # far, empty continue
        x_far = rrg.far(x_rand, eta1, eta2)
        if not x_far:
            continue

        isss = {}
        label_dic = {}
        for x in x_far:
            # steer
            x_new = rrg.steer(x, x_rand)
            # x_new = (0.8, 0.4)
            # label
            label = []
            o_id = True
            x_new2 = rrg.sglp2mulp(x_new)
            for i in range(rrg.robot):
                l = rrg.label(x_new2[i])
                # exists one sampled point lies within obstacles
                if 'o' in l:
                    o_id = False
                    break
                if l != '':
                    l = l + str(i + 1)
                label.append(l)
            if not o_id:
                continue

            # check obstacle free
            isss[(x, x_new)] = rrg.isSimSeg(x, x_new, label)[(x, x_new)]
            if isss[(x, x_new)]:
                added, final1 = construction_p(P, rrg, x, x_new, label, 1)    # x -> x_new
                if added:
                    Xp.append(x_new)
                    Dp.append((x, x_new))
                    label_dic[x_new] = label

        # for x in Xp:
        #     rrg.rrg.add_node(x, label=label_dic[x])
        rrg.rrg.add_edges_from(Dp)

        Dp = []
        for x_new in Xp:
            for x in rrg.near(x_new, eta2):
                try:
                    ss = isss[(x, x_new)]  # (x, x_new) exists
                except KeyError:
                    ss = rrg.isSimSeg(x, x_new, label_dic[x_new])[(x, x_new)]
                if np.linalg.norm( np.subtract(x, rrg.steer(x_new, x))) < rrg.step_size/2 and ss:
                    added, final2 = construction_p(P, rrg, x_new, x, label_dic[x_new], 2) # x_new -> x  close cycle
                    if added:
                        Dp.append((x, x_new))


        rrg.rrg.add_edges_from(Dp)



        if final1:
            length = np.inf
            suf_path = []
            pre_path = nx.algorithms.dijkstra_path(P.p, P.p.graph['init'], final1, weight='dist')
            for succ in P.p.succ[(pre_path[-2][0], final1[1])]:
                len1, suf_path1 = nx.algorithms.single_source_dijkstra(P.p, source = succ, target = (pre_path[-2][0], final1[1]), weight='dist')
                if length > len1 and len(suf_path1) >= 1:
                    suf_path = suf_path1
            # suf_path = nx.algorithms.dijkstra_path(P.p, (pre_path[-2][0], final1[1]), (pre_path[-2][0], final1[1]), weight='dist')
            print(pre_path[:-1], '\n', [(pre_path[-2][0], final1[1])]+ suf_path)
            return pre_path[:-1], [(pre_path[-2][0], final1[1])]+suf_path
        if final2:
            length = np.inf
            suf_path = []
            pre_path = nx.algorithms.dijkstra_path(P.p, P.p.graph['init'], final2, weight='dist')
            for succ in P.p.succ[(pre_path[-2][0], final2[1])]:
                len1, suf_path1 = nx.algorithms.single_source_dijkstra(P.p, source = succ, target = (pre_path[-2][0], final2[1]), weight='dist')
                if length > len1 and len(suf_path1) >= 1:
                    suf_path = suf_path1
            print(pre_path[:-1], '\n', [(pre_path[-2][0], final2[1])]+suf_path)
            return pre_path[:-1], [(pre_path[-2][0], final2[1])]+suf_path


    print(rrg.rrg.number_of_nodes())
    nx.draw(rrg.rrg)
    plt.show()

def construction_p(P, rrg, x, x_new, label, flag):
    """
    Incremental construction of the product automaton
    :param P: product automaton
    :param x: vertex in the ts form: single point
    :param x_new: new state form: single point
    :return: true added
    """
    Sp = []
    Dpp = []
    final = []
    # pair x_new with buchi transition enabled by x
    for node in P.p.nodes():
        if node[0] == x:
            b_state_succ = P.buchi_graph.succ[node[1]]
            for b_state in b_state_succ:
                if P.checkTranB(node[1], rrg.rrg.nodes[x]['label'], b_state):
                    Sp.append((x_new, b_state))
                    Dpp.append((node, (x_new, b_state)))
                    if b_state in P.buchi_graph.graph['accept']:
                        final = (x_new, b_state)

    if Sp:
        # add new node and edge
        if flag == 1:
            rrg.rrg.add_node(x_new, label=label)
        P.p.add_nodes_from(Sp)
        for pair in Dpp:
            P.p.add_edge(pair[0], pair[1], dist = np.linalg.norm(np.subtract(pair[0][0], pair[1][0])))

        # add node and edge due to addition of new node
        x_set = set([node[0] for node in P.p.nodes()])
        while Sp:
            p1 = Sp.pop(0)
            for x2 in x_set:
                if rrg.isSimSeg(x2, p1[0], rrg.rrg.nodes[p1[0]]['label'])[(x2, p1[0])]:
                    for buchi in P.buchi_graph.succ[p1[1]]:
                        if P.checkTranB(p1[1], rrg.rrg.nodes[p1[0]]['label'], buchi):
                            p2 = (x2, buchi)
                            if p2 not in P.p.nodes():
                                P.p.add_node(p2)
                                P.p.add_edge(p1, p2, dist = np.linalg.norm(np.subtract(p1[0], p2[0])))
                                Dpp.append((p1, p2))
                                Sp.append(p2)
                                if buchi in P.buchi_graph.graph['accept']:
                                    final = p2
                            elif (p1, p2) not in P.p.edges():
                                P.p.add_edge(p1, p2, dist = np.linalg.norm(np.subtract(p1[0], p2[0])))
                                Dpp.append((p1, p2))

                else:
                    continue
        return True, final
    return False, final

# +------------------------------------------+
# |     construct transition system graph    |
# +------------------------------------------+
start = datetime.datetime.now()
workspace, regions, obs, init_state, uni_cost, formula = problemFormulation().Formulation()
ts = {'workspace':workspace, 'region':regions, 'obs':obs, 'uni_cost':uni_cost}

n_robot = len(init_state)
step_size = 0.25 * n_robot
rrg = rrg(n_robot, ts, init_state, step_size)


# +------------------------------------------+
# |            construct buchi graph         |
# +------------------------------------------+

buchi = buchi_graph(formula)
buchi.formulaParser()
buchi.execLtl2ba()
buchi_graph = buchi.buchiGraph()

P = P((rrg.mulp2sglp(init_state), buchi_graph.graph['init'][0]), buchi_graph)
pre_path, suf_path = construction_rrg(rrg, P)
for k in range(len(pre_path)):
    pre_path[k] = (rrg.sglp2mulp(pre_path[k][0]), pre_path[k][1])
for k in range(len(suf_path)):
    suf_path[k] = (rrg.sglp2mulp(suf_path[k][0]), suf_path[k][1])

print('Time to find the total path: {0}'.format((datetime.datetime.now() - start).total_seconds()))

path_plot((pre_path, suf_path), regions, obs, rrg.robot, rrg.dim)
plt.show()



