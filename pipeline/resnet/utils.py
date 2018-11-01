import tensorflow as tf

def get_flags():
    flags = tf.app.flags

    flags.DEFINE_string(
            name='logs',short_name='lg',default='./logs/',
            help="The log dir")
    flags.DEFINE_string(
            name='data_dir', short_name='dd', default='/tmp',
            help='The location of the input data.')

    flags.DEFINE_string(
            name='val_dir',short_name='vd',default='/tmp',
            help='The location of the validation data.')

    flags.DEFINE_integer(
            name='num_epochs',short_name='te',default=1,
            help='Number of epochs used to train')

    flags.DEFINE_integer(
            name='batch_size',short_name='bs',default=32,
            help="Batch size for training and evaluation")

    flags.DEFINE_integer(
            name='steps_per_epoch',short_name='spe',default=1000,
            help="Number of steps per epoch")

    flags.DEFINE_string(
            name='model_dir', short_name='md',default='/tmp',
            help="Model save directory")

    flags.DEFINE_string(
            name='labels',short_name='l',default='./mot_labels.txt',
            help="Labels for Dataset")
    
    flags.DEFINE_string(
            name='dtype',short_name='dt',default='float32',
            help='Data type for training (float32 or float16')
    
    return flags
