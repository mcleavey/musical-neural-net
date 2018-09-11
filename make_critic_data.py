from utils import *
from generate import *
import argparse, os, random
from pathlib import Path

PATH = Path('./data/')
TEST=Path("./critic_data/test")
TRAIN=Path("./critic_data/train")

def make_critic_data(num_to_generate, replace, prefix, model_to_load, training, gen_size, use_test_prompt, generator_bs, tt_split):
    PATHS=create_paths()

    # Load pretrained model and training/test text
    lm,params,TEXT=load_pretrained_model(model_to_load, PATHS, training, generator_bs)
    
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
            for mini in range(gen_size//bptt):
                fname=prefix+str(j)+"_"+str(i)+"_"+str(mini)+".txt"
                f=open(dest/'fake'/fname,"w")
                f.write(results[i])
                f.close()
                f=open(dest/'real'/fname,"w")
                f.write(musical_prompts[i][mini*bptt:(mini+1)*bptt])
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
    parser.add_argument("--bs", dest="bs", help="Batch size: # samples to generate (default 64)", type=int)
    parser.set_defaults(bs=64)          
    parser.add_argument("--use_test_prompt", dest="use_test_prompt", action="store_true", help="Use prompt from validation set.")
    parser.set_defaults(use_test_prompt=False)
    parser.add_argument("--test_train_split", dest="tt_split", help="Fraction of test samples (default .1, range 0 to 1", type=float)
    parser.set_defaults(tt_split=.1)    
    args = parser.parse_args()



    random.seed(os.urandom(10))

    make_critic_data(args.num, args.replace, args.prefix, args.model, args.training, args.size, args.use_test_prompt,
          args.bs, args.tt_split)    
