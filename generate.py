from fastai.learner import *
from subprocess import call
import os, argparse
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
from utils import *

import dill as pickle
import random
import numpy as np


PATH = Path('./data/')
MOD_PATH = Path('./models/')
TRAIN = 'train'
VALIDATION = 'test'
OUT = PATH/'output'

    
def load_long_prompts(folder):
    prompts=[]
    all_files=os.listdir(folder)
    for i in range(len(all_files)):
        f=open(folder/all_files[i])
        prompt=f.read()
        prompts.append(prompt)
        f.close()
    return prompts

def music_tokenizer(x): return x.split(" ")
    
def create_generation_batch(model, num_words, random_choice_frequency, 
                            trunc_size, bs, bptt, prompts, params, TEXT):
    prompt_size=bptt
    samples=[]
    offsets=[]
    musical_prompts=[]
    for i in range(bs):
        this_prompt=[]
        timeout=0
        while timeout<100 and len(this_prompt)-prompt_size<=1:
            sample=random.randint(0,len(prompts)-1)
            this_prompt=prompts[sample].split(" ")
            timeout+=1
        assert len(this_prompt)-prompt_size>1, f'After 100 tries, unable to find prompt file longer than {bptt}. Run with smaller --bptt'
            
        offset=random.randint(0, len(this_prompt)-prompt_size-1) 
        samples.append(sample)
        offsets.append(offset)    
        musical_prompts.append(" ".join(this_prompt[offset:prompt_size+offset]))
    

    results=['']*bs
    model.eval()
    model.reset()    
    
    s = [music_tokenizer(prompt)[:bptt] for prompt in musical_prompts]
    t=TEXT.numericalize(s) 

    print("Prompting network")
    for b in t:
        res,*_ = model(b.unsqueeze(0))

    print("Generating new sample")
    for i in range(num_words):
        [ps, n] =res.topk(params["n_tok"])
        
        w=n[:,0]
        for j in range(bs):
            if random.random()<random_choice_frequency:
                ps=ps[:,:trunc_size]
                r=torch.multinomial(ps[j].exp(), 1)
                ind=to_np(r[0])[0]
                if ind!=0:
                    w[j].data[0]=n[j,ind].data[0]
        
            results[j]+=TEXT.vocab.itos[w[j].data[0]]+" "
        res,*_ = model(w.unsqueeze(0))   
    return musical_prompts,results    

def main(model_to_load, training, gen_size, sample_freq, chordwise, 
         note_offset, use_test_prompt, output_folder, generator_bs, trunc, random_freq, prompt_size):

    print("Loading network")     
    lm,params,TEXT=load_pretrained_model(model_to_load, training, generator_bs)
    bptt=prompt_size if prompt_size else params["bptt"]
    
    prompts=load_long_prompts(PATH/VALIDATION) if use_test_prompt else load_long_prompts(PATH/TRAIN)
    print("Preparing to generate a batch of "+str(generator_bs)+" samples.")    
    musical_prompts,results=create_generation_batch(model=lm.model, num_words=gen_size,  
                                                    bs=generator_bs, bptt=bptt,
                                                    random_choice_frequency=random_freq,
                                                    trunc_size=trunc, prompts=prompts,
                                                    params=params, TEXT=TEXT)

    out=OUT/output_folder
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(results)):
        write_to_mp3(results[i], str(i)+".mid", sample_freq, note_offset, out, chordwise)
        fname=str(i)+".txt"
        f=open(out/fname,"w")
        f.write(results[i])
        f.close()
    for i in range(len(musical_prompts)):
        write_to_mp3(musical_prompts[i], "prompt"+str(i)+".mid", sample_freq, note_offset, out, chordwise)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Trained model in ./data/models", required=True)
    parser.add_argument("-output", help="Folder inside ./data/output for holding generations", required=True)

    parser.add_argument("--training", dest="training", help="Trained level (light, med, full, extra). Default: light")
    parser.set_defaults(training="light")
    parser.add_argument("--size", dest="size", help="Number of steps to generate (default 2000)", type=int)
    parser.set_defaults(size=2000)  
    parser.add_argument("--bs", dest="bs", help="Batch size: # samples to generate (default 16)", type=int)
    parser.set_defaults(bs=16)     
    parser.add_argument("--trunc", dest="trunc", help="Truncate guesses to top n (default 5)", type=int)
    parser.set_defaults(trunc=5) 
    parser.add_argument("--random_freq", dest="random_freq", help="How frequently to sample random note (0-1, default .5)", type=float)
    parser.set_defaults(random_freq=.5)     
    parser.add_argument("--sample_freq", dest="sample_freq", help="Split beat into 4 or 12 parts (default 4 for Chordwise, 12 for Notewise)", type=int)
    parser.add_argument("--chordwise", dest="chordwise", action="store_true", help="Use chordwise encoding (defaults to notewise)")
    parser.set_defaults(chordwise=False) 
    parser.add_argument("--small_note_range", dest="small_note_range", action="store_true", help="Set 38 note range (defaults to 62)")
    parser.set_defaults(small_note_range=False)    
    parser.add_argument("--use_test_prompt", dest="use_test_prompt", action="store_true", help="Use prompt from validation set.")
    parser.set_defaults(use_test_prompt=False)
    parser.add_argument("--prompt_size", dest="prompt_size", help="Set prompt size (default is model bptt)", type=int)    
    args = parser.parse_args()

    if args.sample_freq is None:
        sample_freq=4 if args.chordwise else 12
    else:
        sample_freq=args.sample_freq  

    note_offset=45 if args.small_note_range else 33

    random.seed(os.urandom(10))

    main(args.model, args.training, args.size, sample_freq, args.chordwise,
         note_offset, args.use_test_prompt, args.output, args.bs,
         args.trunc, args.random_freq, args.prompt_size)