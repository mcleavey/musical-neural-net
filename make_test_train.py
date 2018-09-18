import os, shutil
import argparse
from pathlib import Path
import random
from tqdm import tqdm




def change_rests(filestr):
    # Chordwise: Without rest modification, model doesn't train well since ~40% of all chords
    # are rests. Instead of listing a rest as 00000000000, I set it according to how many 
    # notes were in the previous 10 time steps. (So if there was one note in the previous 10,
    # the code is bbbbbbbbb, and if there were 8, it would be iiiiiiiiiiii.) 
    # This way the model doesn't learn to do well by always predicting 000000 all the time, 
    # and in addition, it needs to learn some meaningful structure in order to predict the
    # correct rest.
    single_rest="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    notes=filestr.split(" ")
    if len(notes)==0:
        print("String is empty, returning")
        return
    noterange=len(notes[0])-1
    i=0
    while i < len(notes):
        j=1
        if notes[i][1:]=="0"*noterange:
            num_prev_notes=0
            for j in range(-10,0):
                if i+j>=0:
                    if notes[i+j]=="":
                        continue
                    for c in notes[i+j][1:]:
                        if c=="1":
                            num_prev_notes+=1
            num_prev_notes=min(num_prev_notes, len(single_rest)-1)
            notes[i]=notes[i][0]+single_rest[num_prev_notes]*noterange
        i=i+1
    return " ".join(notes)


def remove_duration(DIR):
    # Currently, for Chordwise, I don't use note durations. In 
    # midi-to-encoding I keep them there (0 marks no note, 1 marks the 
    # start of the note, and 2 marks the end of a note). Here I remove the 2s.
    # I don't currently have a good way to handle note durations when predicting
    # Chordwise (probably there needs to be a parallel model predicting the duration for
    # each note in the predicted chord). Directly using the 2s makes the vocab size
    # too large.
    for file in tdqm(os.listdir(DIR)):
        f=open(DIR/file, "r+")
        temp=f.read()
        f.close()
        os.unlink(DIR/file)

        for i in range(12):
            fname=file[:-4]+"___"+str(i)+".txt"
            f=open(DIR/fname, "w")
            f.write(" ".join(temp[i*piece_len:(i+1)*piece_len]))
            f.close()

def remove_wait(DIR):
    # Currently, for Chordwise, I don't use note durations. In 
    # midi-to-encoding I keep them there (0 marks no note, 1 marks the 
    # start of the note, and 2 marks the end of a note). Here I remove the 2s.
    # I don't currently have a good way to handle note durations when predicting
    # Chordwise (probably there needs to be a parallel model predicting the duration for
    # each note in the predicted chord). Directly using the 2s makes the vocab size
    # too large.
    print("Using directory:"+str(DIR))
    files=os.listdir(DIR)
    condense_octaves = False   # Experimented to see if representing octaves as a single token would
                               # help. Initial results were ok but not great, made it more difficult for
                               # model to learn voice leading.
    for j in tqdm(range(len(files))):
        f=open(DIR/files[j], "r+")
        temp=f.read()
        f.close()
        os.unlink(DIR/files[j])
        temp=temp.split(" ")
        
        i=1
        while i<len(temp):
#            print(" ".join(temp[i-1:i+5]))
            if temp[i][:4]=="wait" and temp[i-1][:4]!="wait" and temp[i-1][-3:]!="eoc":
                temp[i-1]=temp[i-1]+"eoc"
                if temp[i]=="wait1":
                    temp[i]=""
                else:
                    temp[i]="wait"+str(int(temp[i][4:])-1)
            elif condense_octaves and (temp[i][:1]=="p" or temp[i][:4]=="endp"):
                k=1
                while i+k<len(temp):
                    if temp[i+k][:4]=="wait":
                        break
                    if temp[i][:1]=="p" and temp[i+k][:1]=="p":
                        if int(temp[i][1:])+12==int(temp[i+k][1:]):
                            temp[i]="p_octave_"+temp[i][1:]
                            temp[i+k]=""
                            break
                    if temp[i][:4]=="endp" and temp[i+k][:4]=="endp":
                        if int(temp[i][4:])+12==int(temp[i+k][4:]):
                            temp[i]="endp_octave_"+temp[i][4:]
                            temp[i+k]=""
                            break                        
                    k+=1                    
            i+=1

        piece_len=len(temp)//12
        for i in range(12):
            fname=files[j][:-4]+"___"+str(i)+".txt"
            f=open(DIR/fname, "w")
            f.write(" ".join(temp[i*piece_len:(i+1)*piece_len]))
            f.close()            
            

def clear_prev(TARGET_TRAIN, TARGET_TEST):
    TARGET_TRAIN.mkdir(parents=True, exist_ok=True)
    TARGET_TEST.mkdir(parents=True, exist_ok=True)    
    for f in os.listdir(TARGET_TRAIN):
        os.unlink(TARGET_TRAIN/f)
    for f in os.listdir(TARGET_TEST):
        os.unlink(TARGET_TEST/f)  

        
def example(TARGET_TRAIN, TARGET_TEST, EXAMPLE_TRAIN, EXAMPLE_TEST):     
    shutil.rmtree(TARGET_TRAIN, ignore_errors=True)
    shutil.copytree(EXAMPLE_TRAIN, TARGET_TRAIN)
    shutil.rmtree(TARGET_TEST, ignore_errors=True)
    shutil.copytree(EXAMPLE_TEST, TARGET_TEST)

        
def main(SOURCE, TARGET_TRAIN, TARGET_TEST, composers, tt_split, chordwise, sample, no_wait):
    """ Clears the current train/test folders and copies new files there
    Input:
        SOURCE - path to original copies of the text files 
        TARGET_TRAIN - data/train
        TARGET_TEST - data/test
        composers - list of composers to include in the train/test creation 
        tt_split - test/train split (.1 = 10% test, 90% train)
        chordwise - bool: use chordwise encoding?  (if not, use notewise encoding)
        sample - use only a subset of the data (.5 = use 50% of all available data)
    Output:
        Copies of files in data/train and data/test, ready for use by train.py
    """
    
    clear_prev(TARGET_TRAIN, TARGET_TEST)
      
    for c in composers:
        print(SOURCE/c)
        files=os.listdir(SOURCE/c)
        for f in files:
            if random.random()>sample:
                continue
            if random.random()<tt_split:
                shutil.copy(SOURCE/c/f, TARGET_TEST)
            else:
                shutil.copy(SOURCE/c/f, TARGET_TRAIN)
    if chordwise:
        print("Chordwise encoding: removing duration marks and expanding rests")
        remove_duration(TARGET_TRAIN)
        remove_duration(TARGET_TEST)
    elif no_wait:
        print("Notewise encoding: removing wait1, wait2")
        remove_wait(TARGET_TRAIN)
        remove_wait(TARGET_TEST)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", dest="example", action="store_true", help="Use sample data (ignores all other flags)")
    parser.set_defaults(example=False)
    parser.add_argument("--composer", dest="composer", help="Composer name (defaults to all). Use bach,brahms,beethoven for multiple composers.") 
    parser.add_argument("--sample_freq", dest="sample_freq", help="Split beat into 4 or 12 parts (default 4 for Chordwise, 12 for Notewise)")
    parser.add_argument("--chordwise", dest="chordwise", action="store_true", help="Use chordwise encoding (defaults to notewise)")
    parser.set_defaults(chordwise=False) 
    parser.add_argument("--no_wait", dest="no_wait", action="store_true", help="Re-encode wait1 markers (notewise only)")
    parser.set_defaults(no_wait=False)     
    parser.add_argument("--chamber", dest="chamber", action="store_true", help="Chamber music (defaults to piano solo)")
    parser.set_defaults(chamber=False) 
    parser.add_argument("--small_note_range", dest="small_note_range", action="store_true", help="Set 38 note range (defaults to 62)")
    parser.add_argument("--train_out", dest="train_out")
    parser.add_argument("--test_out", dest="test_out")
    parser.add_argument("--test_train_split", dest="tt_split", help="Fraction of files to go to test folder (default .1).", type=float)
    parser.set_defaults(tt_split=.1)    
    parser.add_argument("--sample", dest="sample", help="Fraction of files to include: allows smaller sample set for faster training (range 0-1, default 1)", type=float)
    parser.set_defaults(sample=1)
    args = parser.parse_args()

    TARGET_TRAIN=Path('./data/train')
    TARGET_TEST=Path('./data/test')
    EXAMPLE_TRAIN=Path('./data/example_data/train')
    EXAMPLE_TEST=Path('./data/example_data/test')
    
    if args.example:
        example(TARGET_TRAIN,TARGET_TEST,EXAMPLE_TRAIN,EXAMPLE_TEST)
        exit()

    ensemble='chamber' if args.chamber else 'piano_solo'
    encoding='chordwise' if args.chordwise else 'notewise'
    note_range='note_range38' if args.small_note_range else 'note_range62'
    if args.sample_freq is None:
        sample_freq='sample_freq4' if args.chordwise else 'sample_freq12'
    else:
        sample_freq='sample_freq'+str(args.sample_freq)
    SOURCE=Path('./data/composers/')
    SOURCE=SOURCE/encoding/ensemble/note_range/sample_freq
    print(SOURCE)
    
    if args.composer is not None:
        composer=args.composer.split(",")
    else:
        composer=os.listdir(SOURCE)
    

    
    main(SOURCE, TARGET_TRAIN, TARGET_TEST, composer, args.tt_split, args.chordwise, args.sample, args.no_wait)
    
    
    # TODO: ADD assertions about the args (ie, tt_split within 0-1, sample_freq corresponds with files)