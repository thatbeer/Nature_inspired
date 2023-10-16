import xml.etree.ElementTree as ET
import numpy as np


# function to load weight metrics
def load_metrics(xml_path, info=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # meta-data
    name = root.find('name').text
    source = root.find('source').text
    description = root.find('description').text
    doublePrecision = int(root.find('doublePrecision').text)
    ignoredDigits = int(root.find('ignoredDigits').text)
    num_node = len(root.findall(".//vertex"))
    weights_metric = np.nan_to_num(np.identity(num_node) * -np.inf, 0.0)
    # weights_metric = np.identity(num_node)

    for i, vertex in enumerate(root.findall('.//vertex')):
        for edge in vertex.findall('.//edge'):
            cost = float(edge.get("cost"))
            node = int(edge.text)
            # print(f"line:{i} node:{node}->cost:{cost}")
            # if i == node:
            #     weights_metric[i,node] = -np.Inf
            # else:
            weights_metric[i,node] = cost
    if info is True:
        return weights_metric , (name, source, description, doublePrecision, ignoredDigits)

    return weights_metric

