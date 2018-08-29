from classifier_utils import *
import dill as pickle
import argparse

PATH = Path('./data/')
COMPOSER_PATH = Path('./composer_data/')
LOAD_MOD_PATH = Path('./models/generator/')
OUT_MOD_PATH = Path('./models/composer/')

OUT_MOD_PATH.mkdir(parents=True, exist_ok=True)

def train(model, training, use_pretrain, epochs, bs):        
    TEXT=pickle.load(open(f'{LOAD_MOD_PATH}/{model}_text.pkl','rb'))    
    params=pickle.load(open(f'{LOAD_MOD_PATH}/{model}_params.pkl','rb'))
    
    MUSIC_LABEL = data.Field(sequential=False)
    splits = ComposerDataset.splits(TEXT, MUSIC_LABEL, COMPOSER_PATH)
    md = TextData.from_splits(PATH, splits, bs)

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    m = md.get_model(opt_fn, 1500, bptt=params["bptt"], emb_sz=params["em_sz"], n_hid=params["nh"], n_layers=params["nl"], 
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
    m.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    m.clip=25.
    
    if use_pretrain:
        model_name=model+"_"+training
        print("Loading weights: "+model)
        m.load_encoder(f'model_name')
        m.freeze_to(-1)
        
    lrs=[3e-3, 3e-4, 3e-6, 3e-8]
    trainings=["_light.pth", "_med.pth", "_full.pth", "_extra.pth"] 
    save_names=[model_out+b for b in trainings]
    save_names=[OUT_MOD_PATH/s for s in save_names]
        
    for i in range(len(lrs)):
        train_and_save(m, lrs[i], epochs, save_names[i], metrics=[accuracy])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Generator model in ./models/generator", required=True)
    parser.add_argument("--training", help="Training level (light, med, full, extra - default light)")
    parser.set_defaults(training="light")
    parser.add_argument("--no_pretrain", dest="pretrain", action="store_false", help="Start with randomized weights (default starts with generator model weights)")
    parser.set_defaults(pretrain=True)
    parser.add_argument("--epochs", dest="epochs", help="Epochs per training level (default 3)", type=int)
    parser.set_defaults(epochs=3)
    parser.add_argument("--bs", dest="bs", help="Batch size (default 32)", type=int)
    parser.set_defaults(bs=32)

    args = parser.parse_args()


    random.seed(os.urandom(10))

    train(args.model, args.training, args.pretrain, args.epochs, args.bs)        
