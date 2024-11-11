from cProfile import label
import matplotlib.pyplot as plt
import os
import pandas as pd


color_dict = {"LINE": "teal", "DeepWalk": "violet", "Node2Vec": "olive", "ARGA": "orange",
                "PARGA": "blue", "ARVGA": "green", "PARVGA": "red"}

def f_plot(csv_path, data_name, metric="accuracy"):
    all_csv = os.listdir(csv_path)
    all_csv = [_ for _ in all_csv if (("training_" in _) and (data_name.lower() in _) and ("graphsage" not in _))]
    plt.figure(figsize=(10,10))
    plt.xlabel("Epoch", fontdict={'size': 16})
    plt.ylabel("Accuracy", fontdict={'size': 16})
    plt.title(data_name, fontdict={'size': 20})
    all_csv.sort()
    for f in all_csv:
        if "line" in f:
            line_label = "LINE"
        elif "deepwalk" in f:
            line_label = "DeepWalk"
        elif "node2vec" in f:
            line_label = "Node2Vec"
        elif "_ae_" in f:
            if "_alpha0.0_" in f:
                line_label = "ARGA"
            else:
                line_label = "PARGA"
        elif "_vae_" in f:
            if "_alpha0.0_" in f:
                line_label = "ARVGA"
            else:
                line_label = "PARVGA"
        df = pd.read_csv(os.path.join(csv_path, f))
        index = list(range(len(df)))
        accuracy = df.loc[:, metric]
        accuracy_max = df.loc[:, metric+"_max"]
        accuracy_min = df.loc[:, metric+"_min"]
        # plot
        plt.plot(index, accuracy, label=line_label, color=color_dict[line_label])
        plt.fill_between(index, accuracy_min, accuracy_max, alpha=0.15, color=color_dict[line_label])
    plt.legend(loc="lower right") 
    # plt.show()
    plt.savefig("compare_model_%s"%data_name)
    plt.clf()
        
# f_plot("./plot_csv/", "Cora")
# f_plot("./plot_csv/", "Citeseer")
f_plot("./plot_csv/", "Pubmed")