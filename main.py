from configs.training_hyperparameters_config import CFG
from data.dataset import DatasetLoader
from models.multimodel import Hatespeech


def run():
    model = Hatespeech()
    model.load_dataset()
    model.create()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
