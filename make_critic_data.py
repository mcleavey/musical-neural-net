from utils import *
from generate import *
import argparse, os, random
from pathlib import Path

PATH = Path('./data/')
TEST=Path("./critic_data/test")
TRAIN=Path("./critic_data/train")

def main(num_to_generate, replace, prefix, model_to_load, training, gen_size, sample_freq, chordwise, chamber, 
         note_offset, use_test_prompt, generator_bs, tt_split):
    
    # Load pretrained model and training/test text
    lm,params,TEXT=load_pretrained_model(model_to_load, training, generator_bs)
    
    # Load all prompts according to if user wants to use training set or test set prompts
    prompts=load_long_prompts(PATH/"test") if use_test_prompt else load_long_prompts(PATH/"train")

    # Create critic test/train directories if they don't already exist. Clear them if --replace
    for a in [TEST,TRAIN]:
        real=a/'real'
        fake=a/'fake'
        real.mkdir(parents=True, exist_ok=True)
        fake.mkdir(parents=True, exist_ok=True)
        if replace:
            for f in os.listdir(real):
                os.unlink(real/f)
            for f in os.listdir(fake):
                os.unlink(fake/f)   

    # Loop: generate examples and write to file (do this as many times as needed to reach num_to_generate)
    num_iter=num_to_generate//generator_bs+1
    for j in range(0, num_iter):
        print(f'Generating {j} of {num_iter}')
        
        # Generates a batch of prompts and results, each time with a randomized random_choice_frequency
        # and trunc_size (to get a variety of different kinds of generated outputs)
        musical_prompts,results=create_generation_batch(model=lm.model, num_words=gen_size,  
                                                    bs=generator_bs, bptt=gen_size,
                                                    random_choice_frequency=random.random(),
                                                    trunc_size=random.randint(1,10), prompts=prompts,
                                                    params=params, TEXT=TEXT)

        # Write to train/real and train/fake, or test/real and test/fake 
        # Choose randomly whether train or test, according to the test_train_split (tt_split) frequency
        dest=TEST if random.random()<tt_split else TRAIN
        for i in range(generator_bs):
            fname=prefix+str(j)+str(i)+".txt"
            f=open(dest/'fake'/fname,"w")
            f.write(results[i])
            f.close()
            f=open(dest/'real'/fname,"w")
            f.write(musical_prompts[i])
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Trained model in ./data/models", required=True)
    parser.add_argument("-num", help="Number of files to generate (default 1000)", type=int)
    parser.set_defaults(num=1000)
    parser.add_argument("--replace", dest="replace", action="store_true", help="Overwrite existing test/train critic data")
    parser.set_defaults(replace=False)
    parser.add_argument("--prefix", dest="prefix", help="Prefix for output txt files (default None)")
    parser.set_defaults(prefix="")
    parser.add_argument("--training", dest="training", help="Trained level (light, med, full, extra). Default: light")
    parser.set_defaults(training="light")
    parser.add_argument("--size", dest="size", help="Number of steps to generate (default 2000)", type=int)
    parser.set_defaults(size=2000)  
    parser.add_argument("--bs", dest="bs", help="Batch size: # samples to generate (default 16)", type=int)
    parser.set_defaults(bs=16)        
    parser.add_argument("--sample_freq", dest="sample_freq", help="Split beat into 4 or 12 parts (default 4 for Chordwise, 12 for Notewise)", type=int)
    parser.add_argument("--chordwise", dest="chordwise", action="store_true", help="Use chordwise encoding (defaults to notewise)")
    parser.set_defaults(chordwise=False) 
    parser.add_argument("--chamber", dest="chamber", action="store_true", help="Chamber music (defaults to piano solo)")
    parser.set_defaults(chamber=False) 
    parser.add_argument("--small_note_range", dest="small_note_range", action="store_true", help="Set 38 note range (defaults to 62)")
    parser.set_defaults(small_note_range=False)    
    parser.add_argument("--use_test_prompt", dest="use_test_prompt", action="store_true", help="Use prompt from validation set.")
    parser.set_defaults(use_test_prompt=False)
    parser.add_argument("--test_train_split", dest="tt_split", help="Fraction of test samples (default .1, range 0 to 1", type=float)
    parser.set_defaults(tt_split=.1)    
    args = parser.parse_args()

    # Defaults (chordwise usually uses sample_freq 4, notewise uses sample_freq 12
    #           unless the user specifies otherwise)
    
    if args.sample_freq is None:
        sample_freq=4 if args.chordwise else 12
    else:
        sample_freq=args.sample_freq  

    note_offset=45 if args.small_note_range else 33

    random.seed(os.urandom(10))

    main(args.num, args.replace, args.prefix, args.model, args.training, args.size, sample_freq, args.chordwise,
         args.chamber, note_offset, args.use_test_prompt, args.bs, args.tt_split)    
