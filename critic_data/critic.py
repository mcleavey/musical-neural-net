from critic_utils import *
import dill as pickle

PATH = Path('./data/')
MOD_PATH = PATH/'models'

def train():        
    TEXT=pickle.load(open(f'{MOD_PATH}/{model}_text.pkl','rb'))    
    
    MUSIC_LABEL = data.Field(sequential=False)
    splits = MusicDataset.splits(TEXT, MUSIC_LABEL, CRITIC_PATH)
    md = TextData.from_splits(PATH, splits, bs)

    opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    m = md.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl, 
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
    m.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    m.clip=25.
    
    if use_pretrain:
        print("Loading weights: "+model)
        learner.model.load_state_dict(torch.load(MOD_PATH/model))   
        m.freeze_to(-1)
        
    lrs=[3e-3, 3e-4, 3e-6, 3e-8]
    trainings=["_light.pth", "_med.pth", "_full.pth", "_extra.pth"] 
    save_names=[model_out+b for b in trainings]
    save_names=[OUT/s for s in save_names]
        
    for i in range(len(lrs)):
        train_and_save(learner, lrs[i], epochs, save_names[i], metrics=[accuracy])

    
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

best_music, best_eval, worst_music, worst_eval=best_and_worst(1)
fbest="smallnet_best14.mid"
write_to_mp3(best_music_total, fbest, 12)
fbestb="smallnet_best14b.mid"
write_to_mp3(best_music_real, fbestb, 12)
fworst="smallnet_worst14.mid"
write_to_mp3(worst_music_total, fworst, 12)
fworstb="smallnet_worst14b.mid"
write_to_mp3(worst_music_real, fworstb, 12)
