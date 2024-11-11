from constructor import get_placeholder, get_model, format_data, get_optimizer, update
import tensorflow as tf
from sklearn.cluster import KMeans
from metrics import clustering_metrics
import os
import pandas as pd

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class Clustering_Runner():
    def __init__(self, settings):

        print("Clustering on dataset: %s, model: %s, number of iteration: %3d, p_alpha: %4f" % (
            settings['data_name'], settings['model'], settings['iterations'], settings['p_alpha']))

        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.n_clusters = settings['clustering_num']
        self.early_stop_iter = 10
        self.p_alpha = settings['p_alpha']
        self.log_name = "{}_{}_{}".format(
            self.model.split("_")[1], self.data_name, self.p_alpha)

    def erun(self):
        model_str = self.model

        # formatted data
        feas = format_data(self.data_name)

        # Define placeholders
        placeholders = get_placeholder(feas['adj'])

        # construct model
        d_real, discriminator, ae_model = get_model(
            model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders,
                            feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'], self.p_alpha)

        # Initialize session
        sess = tf.compat.v1.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train model
        for epoch in range(1, self.iteration+1):
            emb, cost = update(
                ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
            # print loss
            if epoch % 1 == 0:
                print('epoch = {}, cost = {}'.format(epoch, cost))
                # fh = open('log_{}.txt'.format(self.log_name), 'a')
                # fh.write('epoch = {}, cost = {}'.format(epoch, cost))
                # fh.write('\r\n')
                # fh.flush()
                # fh.close()
            # print cluster result
            if epoch % 1 == 0:
                kmeans = KMeans(n_clusters=self.n_clusters,
                                random_state=0).fit(emb)
                print("Epoch:", '%04d' % (epoch))
                predict_labels = kmeans.predict(emb)
                cm = clustering_metrics(
                    feas['true_labels'], predict_labels, self.log_name)
                cm.evaluationClusterModelFromLabel()
        df = pd.DataFrame()
        df['y'] = list(feas['true_labels'])
        df['emb'] = [list(emb[i, :]) for i in range(len(emb))]
        csv_name = "./log/{}_{}_alpha{}_epoch{}.csv".format(
            self.model.split("_")[1], self.data_name, self.p_alpha, epoch)
        df.to_csv(csv_name, index=0)
        print(csv_name + " is written !!!")
