from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from . import Experiment


def setup_experiment(model_mdl, dataset_mdl, demo):
    experiment = Evaluate(model_mdl, dataset_mdl, demo)
    return experiment


class Evaluate(Experiment):

    def __init__(self, model_mdl, dataset_mdl, demo):
        self.dataset = dataset_mdl.setup_data_loader(model_mdl.__name__)
        self.model = model_mdl.build_model(self.dataset.n_classes)
        self.weights = "./weights/{model}.{dataset}.hdf5".format(model=model_mdl.__name__,
                                                                 dataset=dataset_mdl.__name__)
        self.demo = demo

    def run(self):
        if self.demo:
            self.model.load_weights(self.weights, by_name=True)
            predictions = self.model.predict_on_batch(self.dataset.data_loader(['demo']))
            print zip(self.dataset.categories, sum(predictions)/predictions.shape[0])
        else:
            predictions = list()
            targets = list()
            for data, target in self.dataset.batch_generator_test():
                one_sample_prediction = self.model.predict_on_batch(data)
                predictions.append(sum(one_sample_prediction)/one_sample_prediction.shape[0])
                targets.append(sum(target)/target.shape[0])
            print("Categorical accuracy: {ca}".format(ca=categorical_accuracy(targets, predictions)))
            print("Top-3 categorical accuracy: {t3ca}".format(
                t3ca=top_k_categorical_accuracy(targets, predictions, k=3)))
