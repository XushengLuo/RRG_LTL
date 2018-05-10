# -*- coding: utf-8 -*-

import subprocess
import os.path
import re
from networkx.classes.digraph import DiGraph

class buchi_graph(object):
    """ construct buchi automaton graph
    Parameter:
        formula: LTL formula specifying task
    """

    def __init__(self, formula):
        self.formula = formula

    def formulaParser(self):
        """replace letter with symbol
        """
        indicator = 'FG'

        if [True for i in indicator if i in self.formula]:
            self.formula.replace('F', '<>').replace('G', '[]')

    def execLtl2ba(self):
        """ given formula, exectute the ltl2ba
        Parameter:
            buchi_str: output string of program ltl2ba  (utf-8 format)
        """

        dirname = os.path.dirname(__file__)
        self.buchi_str = subprocess.check_output(dirname + "/./ltl2ba -f \"" + self.formula + "\"", shell=True).decode("utf-8")

    def buchiGraph(self):
        """parse the output of ltl2ba
        Parameter:
            buchi_graph: Graph of buchi automaton
        """
        # find all states
        state_re = re.compile(r'\n(\w+):\n\t')
        state_group = re.findall(state_re, self.buchi_str)

        # find initial and accepting states
        init = [s for s in state_group if 'init' in s]
        accep = [s for s in state_group if 'accept' in s]

        """
        Format:
            buchi_graph.node = NodeView(('T0_init', 'T1_S1', 'accept_S1'))
            buchi_graph.edges = OutEdgeView([('T0_init', 'T0_init'), ('T0_init', 'T1_S1'),....])
            buchi_graph.succ = AdjacencyView({'T0_init': {'T0_init': {'label': '1'}, 'T1_S1': {'label': 'r3'}}})
        """
        self.buchi_graph = DiGraph(type='buchi', init=init, accept=accep)
        for state in state_group:
            # for each state, find transition relation
            # add node
            self.buchi_graph.add_node(state)
            state_if_fi = re.findall(state + r':\n\tif(.*?)fi', self.buchi_str, re.DOTALL)
            if  state_if_fi:
                relation_group = re.findall(r':: \((.*?)\) -> goto (\w+)\n\t', state_if_fi[0])
                for (label, state_dest) in relation_group:
                    # add edge
                    self.buchi_graph.add_edge(state, state_dest, label=label)

        return self.buchi_graph

