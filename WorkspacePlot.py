import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import networkx as nx

def region_plot(regions, flag, ax):
    """
    plot the workspace
    :param regions: regions
    :param flag: regions or obstacle
    :param ax: figure axis
    :param d: 2D or 3D
    :return: none
    """

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(b=True, which='major', color='k', linestyle='--')
    for key in regions.keys():
        if key[1] == 'b':
            circle = plt.Circle(regions[key][0:-1], regions[key][-1], color='b', fill=(flag!='region'))
            ax.add_artist(circle)
            ax.text(regions[key][0], regions[key][1], r'${}_{}$'.format(key[0][0], key[0][1:]), fontsize=16)
        if key[1] == 'p':
            x = []
            y = []
            patches = []
            for i in range(len(regions[key])-1):
                x0, y0 = intersection(regions[key][i], regions[key][i+1])
                x.append(x0)
                y.append(y0)
            x0, y0 = intersection(regions[key][0], regions[key][-1])
            x.append(x0)
            y.append(y0)
            polygon = Polygon(np.column_stack((x,y)), True)
            patches.append(polygon)
            p = PatchCollection(patches)
            ax.add_collection(p)
            ax.text(np.mean(x), np.mean(y), r'${}_{}$'.format(key[0][0], key[0][1:]), fontsize=16)


def intersection(l1, l2):
    """
    calculate the point of intersection between adjacent edges
    :param l1: line 1
    :param l2: line 2
    :return: point of intersection
    """
    if not l1[0]:
        y = -l1[2]/l1[1]
        x = (-l2[2] - l2[1]*y)/l2[0]
    elif not l2[0]:
        y = -l2[2] / l2[1]
        x = (-l1[2] - l1[1] * y) / l1[0]
    else:
        y = -(l1[2]*l2[0] - l2[2]*l1[0])/(l1[1]*l2[0]-l2[1]*l1[0])
        x = (-l1[2]*l2[0]-l1[1]*l2[0]*y)/(l1[0]*l2[0])
    return (x,y)

def path_plot(path, regions, obs, n_robot, dim):
    """
    plot the optimal path in the 2D and 3D
    :param path: ([pre_path], [suf_path])
    :param regions: regions
    :param obs: obstacle
    :return: none
    """

    for n in range(n_robot):
        ax = plt.figure(n).gca()
        region_plot(regions, 'region', ax)
        region_plot(obs, 'obs', ax)

        # prefix path
        x_pre = np.asarray([point[0][n][0] for point in path[0]])
        y_pre = np.asarray([point[0][n][1] for point in path[0]])
        pre = plt.quiver(x_pre[:-1], y_pre[:-1], x_pre[1:] - x_pre[:-1], y_pre[1:] - y_pre[:-1], color='r',
                         scale_units='xy', angles='xy', scale=1, label='prefix path')

        # suffix path
        x = [point[0][n][0] for point in path[1]]
        y = [point[0][n][1] for point in path[1]]
        # x_suf = np.asarray([x_pre[-2]] + x + [x_pre[-2]])
        # y_suf = np.asarray([y_pre[-2]] + y + [y_pre[-2]])
        x_suf = np.asarray(x + [x_pre[-1]])
        y_suf = np.asarray(y + [y_pre[-1]])
        suf = plt.quiver(x_suf[:-1], y_suf[:-1], x_suf[1:] - x_suf[:-1], y_suf[1:] - y_suf[:-1], color='g',
                         scale_units='xy', angles='xy', scale=1, label='suffix path')

        plt.legend(handles=[pre, suf])
        plt.savefig('path{0}.png'.format(n), bbox_inches='tight', dpi=600)


def layer_plot(tree, opt_path, buchi):
    """
    plot 3D layer graph
    :param tree: tree built by alg
    :param buchi: buchi state
    :return: none
    """
    path = list(nx.dfs_edges(tree))

    # 2D workspace in 3D
    fig = plt.figure(2)
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(path)):
        x_pre = np.array([path[i][0][0][0], path[i][1][0][0]])
        y_pre = np.array([path[i][0][0][1], path[i][1][0][1]])
        z_pre = np.asarray([buchi[path[i][0][1]], buchi[path[i][1][1]]])
        ax.plot(x_pre, y_pre, z_pre, 'b--d', markerfacecolor='None')

    x_pre = np.asarray([point[0][0] for point in opt_path[0]])
    y_pre = np.asarray([point[0][1] for point in opt_path[0]])
    z_pre = np.asarray([buchi[point[1]] for point in opt_path[0]])
    ax.plot(x_pre, y_pre, z_pre, 'r--d')
    # region_plot(regions, 'region', ax)
    # region_plot(obs, 'obs', ax)
    plt.savefig('3D.png', bbox_inches='tight', dpi=600)
    # plt.show()


#
# workspace, regions, obs, init_state, uni_cost, formula = problemFormulation().Formulation()
# region_plot(regions, 'region')
# path = (2.50032615691877, ([((0.8, 0.1), 'T0_init'), ((0.5981724872087247, 0.3377620547972048), 'T0_init'), ((0.49389453454964805, 0.4276146121762987), 'T0_init'), ((0.44573857157429486, 0.5078975412511033), 'T1_S8'), ((0.2408944180045951, 0.7160394591937538), 'T1_S8'), ((0.16881448811963862, 0.45388479130328774), 'T2_S8'), ((0.07373085190437723, 0.20101216973373925), 'T2_S8'), ((0.08908170012726457, 0.39929418348934276), 'accept_S8')], [((0.08908170012726457, 0.39929418348934276), 'accept_S8'), ((0.16156655801415953, 0.7450495775375108), 'T1_S8'), ((0.10289083291471668, 0.4017562120131536), 'T2_S8'), ((0.08025318053203045, 0.2943047234035542), 'T2_S8')]))
# path_plot(path[1])

