from fastai.learner import *
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
from utils import *
import dill as pickle
import argparse



# Unlike language models (which need a tokenizer to recognize don't as similar to 'do not', 
# here I have specific encodings for the music, and we can tokenize directly just by splitting by space.
def music_tokenizer(x): return x.split(" ")
    
def main(model_to_load, training, model_out, test, train, bs, bptt, em_sz, nh, nl, min_freq, dropout_multiplier, epochs):
    """ Loads test/train data, creates a model, trains, and saves it
    Input: 
        model_to_load - if continuing training on previously saved model
        model_out - name for saving model
        bs - batch size
        bptt - back prop through time 
        em_sz - embedding size
        nh - hidden vector size
        nl - number of LSTM layers
        min_freq - ignore words that don't appear at least min_freq times in the corpus
        dropout_multiplier - 1 defaults to AWD-LSTM paper (the multiplier scales all these values up or down)
        epochs - number of cycles between saves 
        
    Output:
        Trains model, and saves under model_out_light, _med, _full, and _extra
        Models are saved at data/models

    """
    
    PATHS=create_paths()
    
    # Check test and train folders have files
    train_files=os.listdir(PATHS["data"]/train)
    test_files=os.listdir(PATHS["data"]/test)
    if len(train_files)<2:
        print(f'Not enough files in {PATHS["data"]/train}. First run make_test_train.py')
        return
    if len(test_files)<2:
        print(f'Not enough files in {PATHS["data"]/test}. First run make_test_train.py, or increase test_train_split')
        return    
        
    
    TEXT = data.Field(lower=True, tokenize=music_tokenizer)
    
    # Adam Optimizer with slightly lowered momentum 
    optimizer_function = partial(optim.Adam, betas=(0.7, 0.99))  
    
    if model_to_load:
        print("Loading network")     
        params=pickle.load(open(f'{PATHS["generator"]}/{model_to_load}_params.pkl','rb'))
        LOAD_TEXT=pickle.load(open(f'{PATHS["generator"]}/{model_to_load}_text.pkl','rb'))
        bptt, em_sz, nh, nl = params["bptt"], params["em_sz"], params["nh"], params["nl"]
    
    FILES = dict(train=train, validation=test, test=test)    
    
    # Build a FastAI Language Model Dataset from the training and validation set
    # Mark as <unk> any words not used at least min_freq times
    md = LanguageModelData.from_text_files(PATHS["data"], TEXT, **FILES, bs=bs, bptt=bptt, min_freq=min_freq)

    if model_to_load:
        print(TEXT==LOAD_TEXT)
        TEXT=LOAD_TEXT
        
    print("\nCreated language model data.")
    print("Vocab size: "+str(md.nt))
        
    # AWD LSTM model parameters (with dropout_multiplier=1, these are the values recommended 
    # by the AWD LSTM paper. For notewise encoding, I found that higher amounts of dropout
    # often worked better)
    print("\nInitializing model")
    learner = md.get_model(optimizer_function, em_sz, nh, nl, dropouti=0.05*dropout_multiplier, 
                           dropout=0.05*dropout_multiplier, wdrop=0.1*dropout_multiplier,
                           dropoute=0.02*dropout_multiplier, dropouth=0.05*dropout_multiplier)        

    # Save parameters so that it's fast to rebuild network in generate.py
    dump_param_dict(PATHS, TEXT, md, bs, bptt, em_sz, nh, nl, model_out)
    
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)    # Applying regularization
    learner.clip=0.3                                          # Clip the gradients    

    if model_to_load:
        model_to_load=model_to_load+"_"+training+".pth"
        learner.model.load_state_dict(torch.load(PATHS["generator"]/model_to_load))   
        
    lrs=[3e-3, 3e-4, 3e-6, 3e-8]
    trainings=["_light.pth", "_med.pth", "_full.pth", "_extra.pth"] 
    save_names=[model_out+b for b in trainings]
    save_names=[PATHS["generator"]/s for s in save_names]
        
    for i in range(len(lrs)):
        train_and_save(learner, lrs[i], epochs, save_names[i])

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", dest="bs", help="Batch Size (default 32)", type=int)
    parser.set_defaults(bs=32)
    parser.add_argument("--bptt", dest="bs", help="Back Prop Through Time (default 200)", type=int) 
    parser.set_defaults(bptt=200)
    parser.add_argument("--em_sz", dest="em_sz", help="Embedding Size (default 400)", type=int) 
    parser.set_defaults(em_sz=400)  
    parser.add_argument("--nh", dest="nh", help="Number of Hidden Activations (default 600)", type=int) 
    parser.set_defaults(nh=600)
    parser.add_argument("--nl", dest="nl", help="Number of LSTM Layers (default 4)", type=int) 
    parser.set_defaults(nl=4)
    parser.add_argument("--min_freq", dest="min_freq", help="Minimum frequencey of word (default 1)", type=int) 
    parser.set_defaults(min_freq=1)  
    parser.add_argument("--epochs", dest="epochs", help="Epochs per training stage (default 3)", type=int) 
    parser.set_defaults(epochs=3)      
    parser.add_argument("--prefix", dest="prefix", help="Prefix for saving model (default mod)") 
    parser.set_defaults(prefix="mod")
    parser.add_argument("--dropout", dest="dropout", help="Dropout multiplier (default: 1, range 0-5.)", type=float) 
    parser.set_defaults(dropout=1)    
    parser.add_argument("--load_model", dest="model_to_load", help="Optional partially trained model state dict")
    parser.add_argument("--training", dest="training", help="If loading model, trained level (light, med, full, extra). Default: light")
    parser.set_defaults(training="light")    
    parser.add_argument("--test", dest="test", help="Specify folder name in data that holds test data (default 'test')")
    parser.add_argument("--train",dest="train", help="Specify folder name in data that holds train data (default 'train')")    
    args = parser.parse_args()

    test = args.test if args.test else "test"
    train = args.train if args.train else "train"
    
    main(args.model_to_load, args.training, args.prefix, test, train, args.bs, args.bptt, args.em_sz,
         args.nh, args.nl, args.min_freq, args.dropout, args.epochs)
