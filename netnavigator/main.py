import onnx
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

def onnx_to_nx_graph(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Create a directed graph using NetworkX
    graph = nx.DiGraph()

    # Iterate through the nodes in the ONNX model and add them to the graph
    for node in tqdm(model.graph.node):
        # Add the node to the graph
        graph.add_node(node.name, op_type=node.op_type)

        # Connect input nodes to this node
        for input_name in node.input:
            graph.add_edge(input_name, node.name)

        # Connect this node to its output(s)
        for output_name in node.output:
            graph.add_edge(node.name, output_name)

    return graph

# Specify the path to your ONNX model file
model_path = "gpt2-10.onnx"

# Convert ONNX model to NetworkX graph
nx_graph = onnx_to_nx_graph(model_path)

# Create a plot of the graph
plt.figure(figsize=(10, 500))  # Adjust the figure size as needed
# Use the 'dot' layout for a top-to-bottom directed acyclic graph
pos = nx.drawing.nx_pydot.graphviz_layout(nx_graph, prog="dot")
nx.draw(nx_graph, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black")

# Save the graph as an image
plt.savefig("onnx_graph.png", format="PNG")

# Show the graph (optional)
# plt.show()
