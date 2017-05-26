from . import Experiment


def setup_experiment(model_mdl, dataset_mdl, demo):
    experiment = Preprocess(model_mdl, dataset_mdl, demo)
    return experiment


class Preprocess(Experiment):

    def __init__(self, model_mdl, dataset_mdl, demo):
        self.dataset = dataset_mdl.setup_data_loader(model_mdl.__name__)
        self.model = model_mdl.build_model(self.dataset.n_classes)
        self.demo = demo

    def run(self):
        if self.demo:
            self.dataset.data_loader(['demo'])
        else:
            # Implicitly call data_loader that will check if there is the need to preprocess files and store features
            for data_ids, category_ids in self.dataset.batch_generator_train(endless=False):
                pass
            for data_ids, category_ids in self.dataset.batch_generator_val():
                pass
            for data_ids, category_ids in self.dataset.batch_generator_test():
                pass
