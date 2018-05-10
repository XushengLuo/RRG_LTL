from networkx import DiGraph

g = DiGraph(init=3)
g.add_node(1)
print(g.nodes())
g.add_node(2)
print(g.nodes())
