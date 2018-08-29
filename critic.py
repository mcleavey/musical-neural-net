from critic_utils import *
from utils import *
import dill as pickle
import argparse

PATH = Path('./data/')
CRITIC_PATH = Path('./critic_data/')
LOAD_MOD_PATH = Path('./models/generator/')
OUT_MOD_PATH = Path('./models/critic/')

OUT_MOD_PATH.mkdir(parents=True, exist_ok=True)

def train(model, training, use_pretrain, epochs, bs):        
    TEXT=pickle.load(open(f'{LOAD_MOD_PATH}/{model}_text.pkl','rb'))    
    params=pickle.load(open(f'{LOAD_MOD_PATH}/{model}_params.pkl','rb'))
    
    MUSIC_LABEL = data.Field(sequential=False)
    splits = MusicDataset.splits(TEXT, MUSIC_LABEL, CRITIC_PATH)
    md = TextData.from_splits(PATH, splits, bs)

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    m = md.get_model(opt_fn, 1500, bptt=params["bptt"], emb_sz=params["em_sz"], n_hid=params["nh"], n_layers=params["nl"], 
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
    m.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    m.clip=25.

    if use_pretrain:
        model_name=model+"_"+training
        print("Loading weights: "+model)
        m.load_encoder(model_name)
        m.freeze_to(-1)
        train_and_save(m, 3e-4, 1, model+"_fine_tune.pth", metrics=[accuracy])
        
    lrs=[3e-3, 3e-4, 3e-6, 3e-8]
    trainings=["_light.pth", "_med.pth", "_full.pth", "_extra.pth"] 
    save_names=[model+b for b in trainings]
    save_names=[OUT_MOD_PATH/s for s in save_names]
        
    for i in range(len(lrs)):
        train_and_save(m, lrs[i], epochs, save_names[i], metrics=[accuracy])


def best_and_worst(num_samples=2, bs=32):
    best_eval=-5
    worst_eval=5
    best_music=""
    worst_music=""
    
    for i in range(num_samples):
        random_choice_frequency=random.random()*.5+.5
        trunc_size=random.randint(2,3)
        prompts,results = create_generation_batch(bs=bs, random_choice_frequency=random_choice_frequency, trunc_size=trunc_size)
        s=[music_tokenizer(result)[:1992] for result in results]
        t=TEXT.numericalize(s) 
        preds=m3.model.forward(t)[0][:][:].data
        print("Generation: "+str(i))

        for j in range(bs):
            score=preds[j][1]-preds[j][2]
            
            if score>best_eval:
                best_eval=score
                best_music=results[j]
    
            if score<worst_eval:
                worst_eval=score
                worst_music=results[j]
                

                
    f=open(str(PATH/'best_worst')+'/best.txt', 'w')
    f.write(best_music)
    f.close()
    
    f=open(str(PATH/'best_worst')+'/worst.txt', 'w')
    f.write(worst_music)
    f.close()
    
    return best_music, best_eval, worst_music, worst_eval

def create_best_worst():
    best_music, best_eval, worst_music, worst_eval=best_and_worst(1)
    fbest="smallnet_best14.mid" 
    write_to_mp3(best_music_total, fbest, 12)
    fbestb="smallnet_best14b.mid"
    write_to_mp3(best_music_real, fbestb, 12)
    fworst="smallnet_worst14.mid"
    write_to_mp3(worst_music_total, fworst, 12)
    fworstb="smallnet_worst14b.mid"
    write_to_mp3(worst_music_real, fworstb, 12)



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
