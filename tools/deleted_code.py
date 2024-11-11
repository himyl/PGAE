# from cogdl import pipeline
# from cogdl.datasets import build_dataset_from_name


# 支持的算法有
# emb_models = ["prone", "netmf", "netsmf", "deepwalk", "line",
#             "node2vec", "hope", "sdne", "grarep", "dngr", "spectral"]
# gnn_models = ["dgi", "mvgrl", "grace", "unsup_graphsage"]

# def get_embedding(models=["line", "deepwalk", "node2vec", "unsup_graphsage"],
#                   datasets=["cora", "citeseer", "pubmed"],
#                   dim_emb=128, return_df=False, save_csv=True):
#     if return_df:
#         emb_dict = {}
#     for model in models:
#         for dataset in datasets:
#             print(f"{model}-{dataset} is running !")
#             data = build_dataset_from_name(dataset).data
#             edge_index = np.array([[x, y] for x, y in zip(
#                 data.edge_index[0].tolist(), data.edge_index[1].tolist())])
#             generator = pipeline("generate-emb", model=model, return_model=True,
#                                  num_features=-1, hidden_size=dim_emb, cpu=True,
#                                  cpu_inference=True, no_test=True)
#             outputs = generator(edge_index, data.x.numpy())
#             outputs = [list(outputs[i, :]) for i in range(len(outputs))]
#             df = pd.DataFrame()
#             df['y'] = data.y.numpy()
#             df['emb'] = outputs
#             if save_csv:
#                 csv_name = f"""./embeddings/{model}_{dataset}_{dim_emb}_feature.csv"""
#                 df.to_csv(csv_name, index=0)
#             print(csv_name + "is done !")
#             if return_df:
#                 emb_dict[f"{model}_{dataset}"] = deepcopy(df)
#     if return_df:
#         return deepcopy(emb_dict)
#     else:
#         return {}
