# config.py

class Config(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 4
    max_epochs = 50
    lr = 0.3
    batch_size = 64
    max_sen_len = 100
    dropout_keep = 0.8