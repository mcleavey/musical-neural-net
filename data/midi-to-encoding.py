import argparse
import random
import os
import numpy as np
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note
import music21
from pathlib import Path

# Midi files come in a variety of formats. Here I try to handle files that
# could work as violin or piano, but aren't listed as such.
VIOLINLIKE=["Violin", "Viola", "Cello", "Violincello", "Violoncello", "Flute", 
            "Oboe", "Clarinet", "Recorder", "Voice", "Piccolo",
            "StringInstrument", "Bassoon", "Horn"]
PIANOLIKE=["Piano", "Harp", "Harpsichord", "Organ", ""]


def assign_instrument(instr):
    # Determine if instrument is Piano-like or Violin-like
    if str(instr) in PIANOLIKE:
        return 0
    elif str(instr) in VIOLINLIKE:
        return 1
    else:
        print("Warning, unknown instrument: "+str(instr))
        return -1
    
def stream_to_chordwise(s, chamber, note_range, note_offset, sample_freq):  
    numInstruments=2 if chamber else 1
    maxTimeStep = floor(s.duration.quarterLength * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, numInstruments, note_range))

    notes=[]
    instrumentID=0
    
    noteFilter=music21.stream.filters.ClassFilter('Note')
    chordFilter=music21.stream.filters.ClassFilter('Chord')

    for n in s.recurse().addFilter(noteFilter):
        if chamber:
            instrumentID=assign_instrument(n.activeSite.getInstrument())
            if instrumentID==-1:
                return []
        notes.append((n.pitch.midi-note_offset, floor(n.offset*sample_freq), floor(n.duration.quarterLength*sample_freq), instrumentID))
        
    for c in s.recurse().addFilter(chordFilter):
        pitchesInChord=c.pitches
        if chamber:
            instrumentID=assign_instrument(n.activeSite.getInstrument())     
            if instrumentID==-1:
                return []

        for p in pitchesInChord:
            notes.append((p.midi-note_offset, floor(c.offset*sample_freq), floor(c.duration.quarterLength*sample_freq), instrumentID))

    for n in notes:
        pitch=n[0]
        while pitch<0:
            pitch+=12
        while pitch>=note_range:
            pitch-=12
        if n[3]==1:      #Violin lowest note is v22
            while pitch<22:
                pitch+=12
                
        score_arr[n[1], n[3], pitch]=1                  # Strike note
        score_arr[n[1]+1:n[1]+n[2], n[3], pitch]=2      # Continue holding note
            
    instr={}
    instr[0]="p"
    instr[1]="v"
    score_string_arr=[]
    for timestep in score_arr:
        for i in list(reversed(range(len(timestep)))):   # List violin note first, then piano note
            score_string_arr.append(instr[i]+''.join([str(int(note)) for note in timestep[i]]))      

    return score_string_arr
    
def add_modulations(score_string_arr):
    modulated=[]
    note_range=len(score_string_arr[0])-1
    for i in range(0,12):
        for chord in score_string_arr:
            padded='000000'+chord[1:]+'000000'
            modulated.append(chord[0]+padded[i:i+note_range])
    return modulated

def chord_to_notewise(long_string, sample_freq):
    translated_list=[]
    for j in range(len(long_string)):
        chord=long_string[j]
        next_chord=""
        for k in range(j+1, len(long_string)):
            if long_string[k][0]==chord[0]:
                next_chord=long_string[k]
                break
        prefix=chord[0]
        chord=chord[1:]
        next_chord=next_chord[1:]
        for i in range(len(chord)):
            if chord[i]=="0":
                continue
            note=prefix+str(i)                
            if chord[i]=="1":
                translated_list.append(note)
            # If chord[i]=="2" do nothing - we're continuing to hold the note
            # unless next_chord[i] is back to "0" and it's time to end the note.
            if next_chord=="" or next_chord[i]=="0":      
                translated_list.append("end"+note)
                      
        if prefix=="p":
            translated_list.append("wait")
        
    i=0
    translated_string=""
    while i<len(translated_list):
        wait_count=1
        if translated_list[i]=='wait':
            while wait_count<=sample_freq*2 and i+wait_count<len(translated_list) and translated_list[i+wait_count]=='wait':
                wait_count+=1
            translated_list[i]='wait'+str(wait_count)
        translated_string+=translated_list[i]+" "
        i+=wait_count
        
    return translated_string

def translate_folder_path(START_PATH, note_range, sample_freq, chamber, composer):
    note_range_folder="note_range"+str(note_range)
    sample_freq_folder="sample_freq"+str(sample_freq)
    directory=START_PATH/"chamber" if chamber else START_PATH/"piano_solo"
    directory=directory/note_range_folder/sample_freq_folder/composer
    directory.mkdir(parents=True, exist_ok=True)
    return directory
    
def translate_piece(fname, composer, chamber, sample_freqs, note_ranges, note_offsets, CHORDWISE_PATH, NOTEWISE_PATH, replace):
    # Check if file has already been done previously:
    if not replace:
        exists=True
        for sample_freq in sample_freqs:
            for note_range in note_ranges:    
                seek_file=fname[:-4]+".txt"
                notewise_directory=translate_folder_path(NOTEWISE_PATH, note_range, sample_freq, chamber, composer)
                chordwise_directory=translate_folder_path(CHORDWISE_PATH, note_range, sample_freq, chamber, composer)
                exists = exists and os.path.isfile(notewise_directory/seek_file)
                exists = exists and os.path.isfile(chordwise_directory/seek_file)
                if not exists:
                    break
            if not exists:
                break
        if exists:
            print("Skipping file: Output files already exist. Use --replace to override this and retranslate everything.")
            return
                
    mf=music21.midi.MidiFile()
    try:
        mf.open(fname)
        mf.read()
        mf.close()
    except:
        print("Skipping file: Midi file has bad formatting")
        return
    if chamber and len(mf.tracks)==1:
        print("Skipping file: Expecting chamber music, but piece has only 1 track")        
        return

    print("Waiting for MIT music21.midi.translate.midiFileToStream()")
    try:
        midi_stream=music21.midi.translate.midiFileToStream(mf)
    except:
        print("Skipping file: music21.midi.translate failed")
        return
    print("Translating stream to encodings")

    for sample_freq in sample_freqs:
        for note_range in note_ranges:
    
            score_string_arr = stream_to_chordwise(midi_stream, chamber, note_range, note_offsets[note_range], sample_freq)
            if len(score_string_arr)==0:
                print("Skipping file: Unknown instrument")
                return
       
            score_string_arr=add_modulations(score_string_arr)

            chordwise_directory=translate_folder_path(CHORDWISE_PATH, note_range, sample_freq, chamber, composer)
            os.chdir(chordwise_directory)            

            f=open(fname[:-4]+".txt","w+")
            f.write(" ".join(score_string_arr))
            f.close()
    
            # Translate to notewise format
            score_string=chord_to_notewise(score_string_arr, sample_freq)

            # Write notewise format to file
            notewise_directory=translate_folder_path(NOTEWISE_PATH, note_range, sample_freq, chamber, composer)
            os.chdir(notewise_directory)

            f=open(fname[:-4]+".txt","w+")
            f.write(score_string)
            f.close()
    print("Success")
    
def main(chamber, composers, replace):
    BASE_PATH=Path(os.path.dirname(os.path.realpath(__file__)))
    PATH=BASE_PATH/'composers'/'midi'
    if chamber:
        PATH=PATH/'chamber'
    else:
        PATH=PATH/'piano_solo'

    if len(composers)==0:
        composers=os.listdir(PATH)

    IN_PATH=[PATH/c for c in composers]
    
    CHORDWISE_PATH=BASE_PATH/'composers'/'chordwise'
    NOTEWISE_PATH=BASE_PATH/'composers'/'notewise'
          
    
    sample_freqs=[4,12]
    note_ranges=[38,62]
    note_offsets={}
    note_offsets[38]=45
    note_offsets[62]=33

    for i in range(len(IN_PATH)):
        files=os.listdir(IN_PATH[i])
        for j in range(len(files)):
            fname=files[j]
            print(f'Processing file {j+1} of {len(files)} in {composers[i]}: {fname}...')
            os.chdir(".."/IN_PATH[i])
            output_folder=composers[i]
            translate_piece(fname, output_folder, chamber, sample_freqs, note_ranges, note_offsets, CHORDWISE_PATH, NOTEWISE_PATH, replace)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chamber', dest='chamber', action='store_true', help="Multiple instrument types")
    parser.set_defaults(chamber=False)  
    parser.add_argument('--composers', dest="composers", help="Specify composers (default is all, separate composers by comma)")
    parser.set_defaults(composers="")                       
    parser.add_argument('--replace', dest="replace", action="store_true", help="Retranslate and replace existing files (defaults to skip)")
    parser.set_defaults(replace=False)
    args = parser.parse_args()


    composers=args.composers.split(",") if args.composers else []

    main(args.chamber, composers, args.replace)