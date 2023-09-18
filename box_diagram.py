# Import libraries
import matplotlib.pyplot as plt
import pandas
import numpy
import plotly.express as px
import os
import json

data = []
labels = []
for dirpath, dirnames, filenames in os.walk("."):
    if dirpath.startswith("./pruebas_") and dirpath.endswith("0"):
        print(dirpath)
        labels.append(dirpath)

labels.sort()
print(labels)


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


num_UAV = 3
num_lines = 0
num_dir_outter = 0
for num_dir, (dirname_entrenamiento, dirname_predictor) in enumerate(pairwise([str(dir_) for dir_ in labels])):
    for UAV_idx in range(0, num_UAV):
        to_insert_e = []
        to_insert_p = []
        f = str(UAV_idx+1) + 'UAV.txt'
        print('--------')
        print(os.path.join(dirname_entrenamiento, f))
        print(os.path.join(dirname_predictor, f))
        f_e_opened = os.path.join(dirname_entrenamiento, f)
        f_p_opened = os.path.join(dirname_predictor, f)
        lines_e = open(f_e_opened).read().splitlines()
        lines_p = open(f_p_opened).read().splitlines()
        for UAV_inner_idx in range(0, UAV_idx+1):
            to_insert_e.append([json.loads(i)[UAV_inner_idx] for i in lines_e])
            to_insert_p.append([json.loads(i)[UAV_inner_idx] for i in lines_p])

        to_insert_e_means = numpy.array(to_insert_e)
        to_insert_p_means = numpy.array(to_insert_p)
        to_insert_e_means = list(numpy.average(to_insert_e_means, axis=0))
        to_insert_p_means = list(numpy.average(to_insert_p_means, axis=0))

        data.append(to_insert_e_means)
        data[UAV_idx] += to_insert_p_means
        num_lines = len(lines_e)
    num_dir_outter = num_dir
indexes_to_insert = [1 + (i // num_lines) for i in
                     range(0, num_lines * 2)]  # two tests, then two types of indexes are necessary
data.append(indexes_to_insert)

data = pandas.DataFrame(data).T
print(data)
# data.rename(columns={idx: str(element) for idx, element in enumerate(labels)}, inplace=True)
data.rename(columns={num_UAV: "UAV"}, inplace=True)
data.rename(columns={idx: idx+1 for idx in range(0, num_UAV)}, inplace=True)
data['UAV'] = data['UAV'].astype(str)
data['UAV'] = data['UAV'].replace(str(1.0), 'DRL')
data['UAV'] = data['UAV'].replace(str(2.0), 'PRED')

fig = px.box(data, color="UAV", title="Resultados de evaluación").update_layout(xaxis_title="UAV",
                                                                                yaxis_title="Recompensa acumulada")

fig.show()
