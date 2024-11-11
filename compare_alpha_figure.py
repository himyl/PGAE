from cProfile import label
import matplotlib.pyplot as plt
import os
import pandas as pd


color_dict = {"α=0.01": "teal", "α=0.05": "violet", "α=0.1": "olive", "α=0.3": "orange",
                "α=0.5": "blue", "α=1.0": "green", "α=5.0": "red"}

def f_plot(csv_path, data_name, metric="accuracy"):
    all_csv_ = os.listdir(csv_path)
    for method in ["_ae_", "_vae_"]:
        all_csv = []
        for _ in all_csv_:
            if any(["_alpha" + k.split("=")[1] in _ for k in color_dict]) \
                and _.endswith("epoch200.csv") and data_name.lower() in _ and method in _:
                all_csv.append(_)
        all_csv = sorted(all_csv, key=lambda x: float(x.split("_")[4][5:]))
        plt.figure(figsize=(10,10))
        plt.xlabel("Epoch", fontdict={'size': 16})
        plt.ylabel("Accuracy", fontdict={'size': 16})
        for f in all_csv:
            if "_vae_" in f:
                plt.title("PARVGA on {} with different α".format(data_name), fontdict={'size': 20})
            elif "_ae_" in f:
                plt.title("PARGA on {} with different α".format(data_name), fontdict={'size': 20})
            line_label = "α=" + f.split("_")[4][5:]
            df = pd.read_csv(os.path.join(csv_path, f))
            index = list(range(len(df)))
            accuracy = df.loc[:, metric]
            accuracy_max = df.loc[:, metric+"_max"]
            accuracy_min = df.loc[:, metric+"_min"]
            print(f, list(accuracy)[-1])
            # plot
            plt.plot(index, accuracy, label=line_label, color=color_dict[line_label])
            # plt.fill_between(index, accuracy_min, accuracy_max, alpha=0.15)
        plt.legend(loc='lower right')
        # plt.show()
        plt.savefig("compare_alpha_{}_{}.png".format(method, data_name))
        plt.clf()
        
f_plot("clf_result/", "Cora")
f_plot("clf_result/", "Citeseer")
f_plot("clf_result/", "Pubmed")


