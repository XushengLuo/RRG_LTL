from networkx import DiGraph
class P(object):
    """
    Incremental construction of product automaton
    """
    def __init__(self, init, buchi_graph):
        """
        initialization of product automaton
        :param init: initial state
        :param buchi_graph: buchi graph
        """
        self.init = init
        self.buchi_graph = buchi_graph
        self.p = DiGraph(type='PA', init=init)
        self.p.add_node(init)

    def checkTranB(self, b_state, x_label, q_b_new):
        """ decide valid transition, whether b_state --L(x)---> q_b_new
             Algorithm2 in Chapter 2 Motion and Task Planning
             :param b_state: buchi state
             :param x_label: label of x
             :param q_b_new buchi state
             :return True satisfied
        """
        # b_state_succ = self.buchi_graph.succ[b_state]
        # # q_b_new is not the successor of b_state
        # if q_b_new not in b_state_succ:
        #      return False

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









