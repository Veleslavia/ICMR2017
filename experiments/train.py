import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from . import Experiment
from settings import INIT_LR, LR_REDUCE_EVERY_K_EPOCH, PATIENCE, NUM_EPOCHS, BATCHSIZE


def setup_experiment(model_mdl, dataset_mdl, demo):
    experiment = Train(model_mdl, dataset_mdl, demo)
    return experiment


class Train(Experiment):

    def __init__(self, model_mdl, dataset_mdl, demo):
        self.dataset = dataset_mdl.setup_data_loader(model_mdl.__name__)
        self.model = model_mdl.build_model(self.dataset.n_classes)
        self.key = "{model}.{dataset}".format(dataset=dataset_mdl.__name__, model=model_mdl.__name__)
        self.demo = demo
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE)
        self.lrs = LearningRateScheduler(lambda epoch_n: INIT_LR / (2**(epoch_n//LR_REDUCE_EVERY_K_EPOCH)))
        self.save_clb = ModelCheckpoint("{epoch:02d}-{val_loss:.2f}"+"{key}.hdf5".format(key=self.key),
                                        monitor='val_loss',
                                        save_best_only=True)
        self.model.summary()

    def run(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        history = self.model.fit_generator(self.dataset.batch_generator_train(),
                                           steps_per_epoch=self.dataset.samples_per_epoch/BATCHSIZE,
                                           epochs=NUM_EPOCHS,
                                           verbose=2,
                                           callbacks=[self.save_clb, self.early_stopping, self.lrs],
                                           validation_data=self.dataset.batch_generator_val(),
                                           validation_steps=self.dataset.nb_val_samples/BATCHSIZE,
                                           class_weight=None,
                                           nb_worker=1)
        pickle.dump(history.history, open("{ne}_{key}.pkl".format(ne=NUM_EPOCHS, key=self.key), 'w'))
