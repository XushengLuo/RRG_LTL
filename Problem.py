class problemFormulation(object):

    def __init__(self):

        # +----------------------------------------------+
        # |                                              |
        # |                 Problem 1                    |
        # |                                              |
        # +----------------------------------------------+


        # +-----+-----+-----+
        # |  l1 |     | l2  |
        # |     +-----+     |
        # |       l4        |
        # |             l3  |
        # |    +-------+    |
        # | l5 |       |    |
        # +----+-------+----+
        # l1: (0.2, 0.8)
        # l2: (0.8, 0.8)
        # l3: (0.8, 0.4)
        # l4: (0.4, 0.4)
        # l5: (0.1, 0.2)

        self.ap = {'l1', 'l2', 'l3', 'l4', 'l5'}
        self.workspace =  (1,1)
        # !! no whitespace in atomic proposation      b:ball s:square
        self.regions = {   ('l1', 'b'): (0.2, 0.8, 0.1),
                           ('l2', 'b'): (0.8, 0.8, 0.1),
                           ('l3', 'b'): (0.8, 0.4, 0.1),
                           ('l4', 'b'): (0.4, 0.4, 0.1),
                           ('l5', 'b'): (0.1, 0.2, 0.1) }
        self.obs =  {   ('o1', 'p'): ((0, 1, -1), (1, 0, -0.6), (0, -1, 0.7), (-1, 0, 0.4)),     # coefficient <=0
                        ('o2', 'p'): ((0, 1, -0.2), (1, 0, -0.7), (0, -1, 0), (-1, 0, 0.3)),
                        # ('o3', 'p'): ((1, 1, -1), (-1, 1, 0.2), (0, -1, 0.2)),
                        # ('o4', 'b'): (0.3, 0.6, 0.1)
                      }
        self.init_state = ((0.8, 0.1),(0.8, 0.1))
        # self.init_state = ((0.2, 0.8), (0.2, 0.8))
        # self.init_state = ((0.8, 0.1),)
        self.uni_cost = 0.1

        # #----------------------------------------------#
        # |                                              |
        # |                 Problem 2                    |
        # |                                              |
        # #----------------------------------------------#

        # +-----+-----+-----+
        # | r4,b|r5,rb| r6  |
        # +-----+-----+-----+
        # | c1  | c2  | c3  |
        # +-----+-----+-----+
        # | r1  | r2,b|r3,gb|
        # +-----+-----+-----+

        # self.ap = {'l1', 'l2', 'l3', 'l4', 'l5'}
        # self.workspace = (1, 1)
        # # !! no whitespace in atomic proposation      b:ball s:square
        # self.regions = {('l1', 'b'): (0.2, 0.8, 0.1),
        #                 ('l2', 'b'): (0.8, 0.8, 0.1),
        #                 ('l3', 'b'): (0.8, 0.4, 0.1),
        #                 ('l4', 'b'): (0.4, 0.4, 0.1),
        #                 ('l5', 'b'): (0.1, 0.2, 0.1)}
        # # self.obs = {('o1', 'p'): ((0, 1, -1), (1, 0, -0.6), (0, -1, 0.7), (-1, 0, 0.4)),  # coefficient <=0
        # #             ('o2', 'p'): ((0, 1, -0.2), (1, 0, -0.7), (0, -1, 0), (-1, 0, 0.3)),
        # #             ('o3', 'p'): ((1, 1, -1), (-1, 1, 0.2), (0, -1, 0.2)),
        # #             ('o4', 'b'): (0.3, 0.6, 0.1)
        # #             }
        # self.obs = {}
        # self.init_state = (0.8, 0.1)
        # self.uni_cost = 0.1


        """
        +----------------------------+
        |   Propositonal Symbols:    |
        |        true, false         |
        |	    any lowercase string |
        |                            |
        |   Boolean operators:       |
        |       !   (negation)       |
        |       ->  (implication)    |
        |	    <-> (equivalence)    |
        |       &&  (and)            |
        |       ||  (or)             |
        |                            |
        |   Temporal operators:      |
        |       []  (always)         |
        |       <>  (eventually)     |
        |       U   (until)          |
        |       V   (release)        |
        |       X   (next)           |
        +----------------------------+
        """

        # self.formula = '<>(l3 && []<>l4)'
        # self.formula = '<>l41 && []<>l11 && [](l11 -> X(!l11 U l21)) '
        # self.formula = '<>l41 && []<>l11 && [](l11 -> X(!l11 U l21)) && []<>l31 && []!l51'
        self.formula = '[]<>l11 && [](l11 -> X(!l11 U l22))'
        # self.formula = '[]<>l11 && []<>l12'
        # self.formula = '[]<>l11 && []<>l12'
        # self.formula = '<>l4 && []<>l1 && [](l1 -> X(!l1 U l2)) && []<>l3 && []!l5'
        #self.formula = '<>l4 && []<>l1 && []<>l5'  # formula 1
        #self.formula =  '<>(rb && <>b) && <>[]r1 && [](rb -> X(!gb U b)) && <>(gb && <>b) && [](gb -> X(!rb U b))'  # \phi 2
        #self.formula = '<>(rb && <>(b && r2)) && <>[]r1 && [](rb -> X(!gb U b)) && <>(gb && <>(b && r4)) && [](gb -> X(!rb U b))'   #\phi 3
        #self.formula = '([]<>r4) && ([]<>r3) && ([]<>r6)'     # \phi 4 inspect room r3, r4, r6 infinitely often
        #self.formula = 'gb U b'
    def Formulation(self):
        print('Task specified by LTL formula: ' + self.formula)
        return self.workspace, self.regions, self.obs, self.init_state, self.uni_cost, self.formula