import os
import dill as pickle
from pathlib import Path
import random
import numpy as np
import pandas as pd
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
import random
from fastai.learner import *
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
import dill as pickle

PATH = Path('./data/')
MOD_PATH = PATH/'models'

def train_and_save(learner, lr, epochs, fname, metrics=None):
    print("Training "+str(fname))
    learner.fit(lr, 1, wds=1e-6, cycle_len=epochs, use_clr=(32,10), metrics=metrics)
    print("Saving "+str(fname))
    torch.save(learner.model.state_dict(), fname)

def load_pretrained_model(model_to_load, training, bs):
    params=pickle.load(open(f'{MOD_PATH}/{model_to_load}_params.pkl','rb'))
    TEXT=pickle.load(open(f'{MOD_PATH}/{model_to_load}_text.pkl','rb'))
    lm = LanguageModel(to_gpu(get_language_model(params["n_tok"], params["em_sz"], params["nh"], 
                                                    params["nl"], params["pad"])))

    mod_name=model_to_load+"_"+training+".pth"
    lm.model.load_state_dict(torch.load(MOD_PATH/mod_name)) 
    lm.model[0].bs=bs
    return lm, params, TEXT

def dump_param_dict(TEXT, md, bs, bptt, em_sz, nh, nl, model_out):
    d={}
    d["n_tok"]=md.nt
    d["pad"]=md.pad_idx
    d["bs"]=bs
    d["bptt"]=bptt
    d["em_sz"]=em_sz
    d["nh"]=nh
    d["nl"]=nl
    
    pickle.dump(d, open(f'{MOD_PATH}/{model_out}_params.pkl','wb'))
    pickle.dump(TEXT, open(f'{MOD_PATH}/{model_out}_text.pkl','wb'))
    
    
def write_midi(s, filename, output_folder):
    fp = s.write('midi', fp=output_folder/filename)
    
def string_inds_to_stream(string, sample_freq, note_offset, chordwise):
    score_i = string.split(" ")
    if chordwise:
        return arrToStreamChordwise(score_i, sample_freq, note_offset)
    else:
        return arrToStreamNotewise(score_i, sample_freq, note_offset)

def arrToStreamChordwise(score, sample_freq, note_offset):

    speed=1./sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0
    for i in range(len(score)):
        if len(score[i])==0:
            continue

        for j in range(1,len(score[i])):
            if score[i][j]=="1":
                duration=2
                new_note=music21.note.Note(j+note_offset)    
                new_note.duration = music21.duration.Duration(duration*speed)
                new_note.offset=(i+time_offset)*speed
                if score[i][0]=='p':
                    piano_notes.append(new_note)
                elif score[i][0]=='v':
                    violin_notes.append(new_note)
    violin=music21.instrument.fromString("Violin")
    piano=music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    violin_stream=music21.stream.Stream(violin_notes)
    piano_stream=music21.stream.Stream(piano_notes)
    main_stream = music21.stream.Stream([violin_stream, piano_stream])
    return main_stream
                    
def arrToStreamNotewise(score, sample_freq, note_offset):
    speed=1./sample_freq
    piano_notes=[]
    violin_notes=[]
    time_offset=0
    for i in range(len(score)):
        if score[i] in ["", " ", "<eos>", "<unk>"]:
            continue
        elif score[i][:3]=="end":
            continue
        elif score[i][:4]=="wait":
            time_offset+=int(score[i][4:])
            continue
        else:
            # Look ahead to see if an end<noteid> was generated
            # soon after.  Just brute forcing it for now - I'll 
            # come back to redo this if it's too slow
            duration=1
            has_end=False
            for j in range(1,200):
                if i+j==len(score):
                    break
                if score[i+j][:4]=="wait":
                    duration+=int(score[i+j][4:])
                if score[i+j]=="end"+score[i] or score[i+j]==score[i]:
                    has_end=True
                    break
            if not has_end:
                duration=12
                
            new_note=music21.note.Note(int(score[i][1:])+note_offset)    
            new_note.duration = music21.duration.Duration(duration*speed)
            new_note.offset=time_offset*speed
            if score[i][0]=="v":
                violin_notes.append(new_note)
            else:
                piano_notes.append(new_note)
    violin=music21.instrument.fromString("Violin")
    piano=music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    violin_stream=music21.stream.Stream(violin_notes)
    piano_stream=music21.stream.Stream(piano_notes)
    main_stream = music21.stream.Stream([violin_stream, piano_stream])
    return main_stream

def write_to_mp3(stream, fname, sample_freq, note_offset, out, chordwise):
    stream_out=string_inds_to_stream(stream, sample_freq, note_offset, chordwise)
    write_midi(stream_out, fname, out)
    base=out/fname[:-4]
    os.system(f'./data/mid2mp3.sh {base}.mid')
    os.system(f'mpg123 -w {base}.wav {base}.mp3')    
    
    