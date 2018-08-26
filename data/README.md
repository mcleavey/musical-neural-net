midi-to-encoding.py
Different ways of translating from midi file to a format ready for input to the LSTM. As a default, it will run through all the composers in the composers/midi folder, and will output several different kinds of input formats.  

notewise - this outputs each note individually. For example: p24 v18 would be the piano note 24 and violin note 18 played at the same time.
wait indicates all the notes at that musical timestep have been listed and it's time to move to the next step.  wait12 or more generally wait<x> means there will be no more notes for the next <x> musical time steps.  
endv18 means it's time for the violin to stop playing note 18.  A sequence might be: p4 p18 p24 v28 wait4 endp4 p12 endp18 endv28 wait2
    
chordwise - this outputs all the violin notes at a given timestep, followed by all the piano notes at a given timestep. 1 means a note is played on that step. 2 means a note is held. 0 means a note is silend. A sequence might be: p00001001000000000 p0000200000001000

sample_freq is the number of samples per quarter note (to capture 16th notes accurately, it should be at least 4. To capture both 16ths and triplets, it can be set to 12).  Probably a good setting is 12 for notewise and 4 for chordwise.

note_range is the range from lowest note to highest note considered by the model. If a note in the midi file is too high or too low, it is transposed up or down an octave.  Violin notes are not allowed to be below the violin's G string.

data-collector.py
A makeshift script for downloading midi files from classicalarchives.com
This assumes you have already gone through and hand selected the pieces to download (Classical Archives allows 100 downloads per day).  I didn't automate this because I wanted more control over which pieces I was selecting (and wanted to avoid pieces that were arranged for piano 4 hands, or other variations).  

If I have more time, I'll come back and rewrite this so that it includes the username/password login.  For now, I find it faster to go to https://www.classicalarchives.com/secure/downloads.html and click to view page source. I then copy that page source to a txt file in the http_source folder.

Example:
python data-collector.py --source ./http_source/http.txt --composer brahms

