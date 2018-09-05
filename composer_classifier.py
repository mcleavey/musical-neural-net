from classifier_utils import *
from utils import *
import dill as pickle
import argparse


def train(model_to_load, training, use_pretrain, epochs, bs):    
    PATHS=create_paths()
    
    # TEXT holds information about the vocab to index mapping
    # params holds the hyperparameters of the model (embedding size, number layers, etc)
    TEXT=pickle.load(open(f'{PATHS["generator"]}/{model_to_load}_text.pkl','rb'))    
    params=pickle.load(open(f'{PATHS["generator"]}/{model_to_load}_params.pkl','rb'))
    
    print("Loading dataset")
    MUSIC_LABEL = data.Field(sequential=False)
    splits = ComposerDataset.splits(TEXT, MUSIC_LABEL, PATHS["composer_data"])
    music_data = TextData.from_splits(PATHS["data"], splits, bs)

    print("Initializing model")
    optimizer_function = partial(optim.Adam, betas=(0.6, 0.95))  # Adam optimizer with lower momentum
    
    model = music_data.get_model(optimizer_function, 1500, bptt=params["bptt"], emb_sz=params["em_sz"], 
                                 n_hid=params["nh"], n_layers=params["nl"], dropout=0.1, dropouti=0.4,
                                 wdrop=0.5, dropoute=0.05, dropouth=0.3)  # Dropout defaults from AWD paper
    
    model.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)         # Regularization (from FastAI library)
    model.clip=25.                                               # Clipping gradients
    
    if use_pretrain:
        raise NotImplementedError
        
    # Training uses cyclical learning rates. These are the initial weights for 
    # each cycle. The model is saved as _light after the first cycle, as _med after
    # the second cycle, etc. Each cycle is "epochs" long
    
    lrs=[3e-4, 3e-4, 3e-6, 3e-8]
    trainings=["_light.pth", "_med.pth", "_full.pth", "_extra.pth"] 
    save_names=[model_to_load+b for b in trainings]
    save_names=[PATHS["composer"]/s for s in save_names]
        
    for i in range(len(lrs)):
        train_and_save(model, lrs[i], epochs, save_names[i], metrics=[accuracy])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Generator model in ./models/generator", required=True)
    parser.add_argument("--training", help="Training level (light, med, full, extra - default light)")
    parser.set_defaults(training="light")
    parser.add_argument("--pretrain", dest="pretrain", action="store_true", help="Starts with generator model weights (default is random initialization)")
    parser.set_defaults(pretrain=False)
    parser.add_argument("--epochs", dest="epochs", help="Epochs per training level (default 3)", type=int)
    parser.set_defaults(epochs=3)
    parser.add_argument("--bs", dest="bs", help="Batch size (default 32)", type=int)
    parser.set_defaults(bs=32)

    args = parser.parse_args()

    random.seed(os.urandom(10))

    train(args.model, args.training, args.pretrain, args.epochs, args.bs)        
