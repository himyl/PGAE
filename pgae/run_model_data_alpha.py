import settings

import sys

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner


model = sys.argv[1]           # 'arga_ae' or 'arga_vae'
task = 'clustering'         # 'clustering' or 'link_prediction'
early_stop_iter = 10
dataname = sys.argv[2]

settings = settings.get_settings(dataname, model, task)

settings['p_alpha'] = float(sys.argv[3])

if task == 'clustering':
    runner = Clustering_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()

