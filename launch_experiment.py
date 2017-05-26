import importlib
import argparse


def run_experiment(experiment_type_mdl, model_mdl, dataset_mdl, demo):
    experiment = experiment_type_mdl.setup_experiment(model_mdl, dataset_mdl, demo)
    experiment.run()


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-t', '--type', dest='type',
                         required=True, action='store',
                         help='-t type of the experiment, accept strings: preprocess, train, test')
    aparser.add_argument('-m', '--model', dest='model',
                         required=True, action='store',
                         help='-m model for the experiment, depends on the type of the experiment')
    aparser.add_argument('-d', '--dataset', dest='dataset',
                         required=True, action='store',
                         help='-d dataset for the experiment: fcvid or yt8m')
    aparser.add_argument('-o', '--demo_option', dest='demo',
                         required=False, action='store_true', default=False,
                         help='-o recognition demo')

    args = aparser.parse_args()

    experiment_type_mdl = importlib.import_module(".{}".format(args.type), "experiments")
    print "{} imported as 'type'".format(args.type)

    model_mdl = importlib.import_module(".{}".format(args.model), "models")
    print "{} imported as 'model'".format(args.model)

    dataset_mdl = importlib.import_module(".{}".format(args.dataset), "datasets")
    print "{} imported as 'dataset'".format(args.dataset)

    demo = args.demo

    run_experiment(experiment_type_mdl, model_mdl, dataset_mdl, demo)


if __name__ == "__main__":
    main()
