from absl import app
from resnet.box_net import Box_net
from resnet.utils import get_flags
import tensorflow as tf

flags = get_flags()

def main(argv):
    global flags
    cr = Box_net(flags.FLAGS)
    cr.ready_dataset()
    cr.train()


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
