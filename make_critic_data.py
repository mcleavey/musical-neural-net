from utils import *
from generate import *
import argparse, os, pathlib, random

TEST=Path(./critic_data/test)
TRAIN=Path(./critic_data/train)

def main(num_to_generate, model_to_load, training, gen_size, sample_freq, chordwise, chamber, 
         note_offset, use_test_prompt, output_folder, generator_bs, tt_split):
    lm,params,TEXT=load_pretrained_model(model_to_load, training, generator_bs)
    prompts=load_long_prompts(PATH/VALIDATION) if use_test_prompt else load_long_prompts(PATH/TRAIN)

    for a in [TEST,TRAIN]:
    	real=a/'real'
    	fake=a/'fake'
	    real.mkdir(parents=True, exist_ok=True)
	    fake.mkdir(parents=True, exist_ok=True)
    	for f in os.listdir(real):
        	os.unlink(real/f)	
        for f in os.listdir(fake):
        	os.unlink(fake/f)		    

    for j in range(0, num_to_generate//bs + 1):
	    musical_prompts,results=create_generation_batch(model=lm.model, num_words=gen_size,  
                                                    bs=generator_bs, bptt=gen_size,
                                                    random_choice_frequency=random.random(),
                                                    trunc_size=random.randint(1,10), prompts=prompts,
                                                    params=params, TEXT=TEXT)

	    dest=TEST if random.random()<tt_split else TRAIN
	    for i in range(bs):
			fname=str(j)+str(i)+".txt"
	        f=open(dest/'fake'/fname,"w")
    	    f.write(results[i])
        	f.close()
        	f=open(dest/'real'/fname,"w")
    	    f.write(musical_prompts[i])
        	f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Trained model in ./data/models")
    parser.add_argument("-output", help="Folder inside ./data/output for holding generations")

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
    args = parser.parse_args()

    if args.sample_freq is None:
        sample_freq=4 if args.chordwise else 12
    else:
        sample_freq=args.sample_freq  

    note_offset=45 if args.small_note_range else 33

    random.seed(os.urandom(10))

    main(args.model, args.training, args.size, sample_freq, args.chordwise,
         args.chamber, note_offset, args.use_test_prompt, args.output, args.bs)    