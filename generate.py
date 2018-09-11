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


    
def load_long_prompts(folder):
    """ folder is either the path to train or to test
        load_long_prompts loads all the files in that folder and returns 
        a list holding the text inside each file
    """
    prompts=[]
    all_files=os.listdir(folder)
    for i in range(len(all_files)):
        f=open(folder/all_files[i])
        prompt=f.read()
        prompts.append(prompt)
        f.close()
    return prompts

def music_tokenizer(x): return x.split(" ")
    

def generate_musical_prompts(prompts, bptt, bs):
    prompt_size=bptt
    musical_prompts=[]
    
    # Randomly select bs different prompts and hold them in musical_prompts
    for i in range(bs):
        this_prompt=[]
        timeout=0
        while timeout<100 and len(this_prompt)-prompt_size<=1:
            sample=random.randint(0,len(prompts)-1)
            this_prompt=prompts[sample].split(" ")
            timeout+=1
        assert len(this_prompt)-prompt_size>1, f'After 100 tries, unable to find prompt file longer than {bptt}. Run with smaller --bptt'
            
        offset=random.randint(0, len(this_prompt)-prompt_size-1)     
        musical_prompts.append(" ".join(this_prompt[offset:prompt_size+offset]))

    return musical_prompts


def create_generation_batch(model, num_words, random_choice_frequency, 
                            trunc_size, bs, bptt, prompts, params, TEXT):
    """ Generate a batch of musical samples
    Input:
      model - pretrained generator model
      num_words - number of steps to generate
      random_choice_frequency - how often to pick a random choice rather than the top choice (range 0 to 1)
      trunc_size - for the random choice, cut off the options to include only the best trunc_size guesses (range 1 to vocab_size)
      bs - batch size - number of samples to generate
      bptt - back prop through time - size of prompt
      prompts - a list of training or test folder texts
      params - parameters of the generator model
      TEXT - holds vocab word to index dictionary   
     
    Output:
      musical_prompts - the randomly selected prompts that were used to prime the model (these are human-composed samples)
      results - the generated samples
      
    This is very loosely based on an example in the FastAI notebooks, but is modified to include randomized prompts,
    to generate a batch at a time rather than a single example, and to include truncated random sampling.
    """

    musical_prompts=generate_musical_prompts(prompts, bptt, bs)

    results=['']*bs
    model.eval()
    model.reset()    

    # Tokenize prompts and translate them to indices for input into model
    s = [music_tokenizer(prompt)[:bptt] for prompt in musical_prompts]
    t=TEXT.numericalize(s) 

    print("Prompting network")
    # Feed the prompt one by one into the model (b is a vector of all the indices for each prompt at a given timestep)
    for b in t:
        res,*_ = model(b.unsqueeze(0))

    print("Generating new sample")
    for i in range(num_words):
        # res holds the probabilities the model predicted given the input sequence
        # n_tok is the number of tokens (ie the vocab size)
        [ps, n] =res.topk(params["n_tok"])
        
        # By default, choose the most likely word (choice 0) for the next timestep (for all the samples in the batch)
        w=n[:,0]
        
        # Cycle through the batch, randomly assign some of them to choose from the top trunc guesses, rather than to
        # automatically take the top choice
        for j in range(bs):
            if random.random()<random_choice_frequency:
                # Truncate to top trunc_size guesses only
                ps=ps[:,:trunc_size]
                # Sample based on the probability the model predicted for those top choices
                r=torch.multinomial(ps[j].exp(), 1)
                # Translate this to an index 
                ind=to_np(r[0])[0]
                if ind!=0:
                    w[j].data[0]=n[j,ind].data[0]

            # Translate the index back to a word (itos is index to string) 
            # Append to the ongoing sample
            results[j]+=TEXT.vocab.itos[w[j].data[0]]+" "

        # Feed all the predicted words from this timestep into the model, in order to get predictions for the next step
        res,*_ = model(w.unsqueeze(0))   
    return musical_prompts,results    

def main(model_to_load, training, test, train, gen_size, sample_freq, chordwise, 
         note_offset, use_test_prompt, output_folder, generator_bs, trunc, random_freq, prompt_size):

    PATHS=create_paths()
    print("Loading network")     
    lm,params,TEXT=load_pretrained_model(model_to_load, PATHS, training, generator_bs)
    bptt=prompt_size if prompt_size else params["bptt"]
    
    prompts=load_long_prompts(PATHS["data"]/test) if use_test_prompt else load_long_prompts(PATHS["data"]/train)
    print("Preparing to generate a batch of "+str(generator_bs)+" samples.")    
    musical_prompts,results=create_generation_batch(model=lm.model, num_words=gen_size,  
                                                    bs=generator_bs, bptt=bptt,
                                                    random_choice_frequency=random_freq,
                                                    trunc_size=trunc, prompts=prompts,
                                                    params=params, TEXT=TEXT)

    # Create the output folder if it doesn't already exist
    out=PATHS["output"]/output_folder
    out.mkdir(parents=True, exist_ok=True)
    
    # For each generated sample, write mid, mp3, wav, and txt files to the output folder (as 1.mid, etc)
    for i in range(len(results)):
        write_mid_mp3_wav(results[i], str(i).zfill(2)+".mid", sample_freq, note_offset, out, chordwise)
        fname=str(i)+".txt"
        f=open(out/fname,"w")
        f.write(results[i])
        f.close()
        
    # For each human-composed sample, write mid, mp3, and wav files to the output folder (as prompt1.mid, etc)
    for i in range(len(musical_prompts)):
        write_mid_mp3_wav(musical_prompts[i], "prompt"+str(i).zfill(2)+".mid", sample_freq, note_offset, out, chordwise)

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
    parser.add_argument("--test", dest="test", help="Specify folder name in data that holds test data (default 'test')")
    parser.add_argument("--train",dest="train", help="Specify folder name in data that holds train data (default 'train')")
    args = parser.parse_args()

    if args.sample_freq is None:
        sample_freq=4 if args.chordwise else 12
    else:
        sample_freq=args.sample_freq  

    note_offset=45 if args.small_note_range else 33

    random.seed(os.urandom(10))

    test = args.test if args.test else "test"
    train = args.train if args.train else "train"
    
    main(args.model, args.training, test, train, args.size, sample_freq, args.chordwise,
         note_offset, args.use_test_prompt, args.output, args.bs,
         args.trunc, args.random_freq, args.prompt_size)