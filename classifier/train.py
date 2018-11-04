from absl import app
from resnet.classifier import Classifier
from resnet.utils import get_flags
import tensorflow as tf

flags = get_flags()

def main(argv):
    global flags
    cr = Classifier(flags.FLAGS)
    cr.ready_dataset()
    cr.train()


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
