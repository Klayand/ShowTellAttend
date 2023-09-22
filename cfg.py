from yacs.config import CfgNode as CN


CFG = CN()

# Path
CFG.ROOT_DIR = f'flickr8k/images'
CFG.ANNOTATION_DIR = f'flickr8k/captions.txt'

# Hyperparameters
CFG.EMBED_SIZE = 256
CFG.HIDDEN_SIZE = 256
CFG.VOCAB_SIZE = 0
CFG.NUM_LAYERS = 5
