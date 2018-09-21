# Clara: A Neural Net Music Generator
Take the <a href="http://christinemcleavey.com/human-or-ai/">AI vs Human Quiz</a>.<br>

Train an <a href="https://arxiv.org/pdf/1708.02182.pdf">AWD-LSTM</a> to generate piano or violin/piano music<br>
Project overview is <a href="http://christinemcleavey.com/clara-a-neural-net-music-generator/">here</a>.<br>
Detailed paper is <a href="http://christinemcleavey.com/files/clara-musical-lstm.pdf">here</a>.<br>

<h2>Requirements:</h2>
<ul>
<li>PyTorch version 0.3.0.post4</li>
<li><a href="https://github.com/fastai/fastai">FastAI</a>
  </li>
<li><a href="http://schristiancollins.com/generaluser.php">GeneralUser</a>: install in data folder (this is needed to translate midi files to mp3)</li>
  <li><a href="http://web.mit.edu/music21/">MIT's music21</a></li>
</ul>
Note: From inside the musical-neural-net home directory, run: 

```
ln -s ./replace/this/with/your/path/to/fastai/library fastai 
```

to create a symbolic link to the fastai library. Alternately, <a href="https://medium.com/@youngladesh/setup-and-run-fast-ai-in-amazon-aws-7fd028351a1e">this blog</a> has a clear description of how to get an AWS machine up and running with FastAI already good to go.<br><br>
You will also likely need to use sudo apt install to get fluidsynth, mpg321, and twolame.

<h2>Basic:</h2>
Run the Jupyter Notebook BasicIntro.ipynb or follow the individual instructions here. 
To create generations with a pretrained notewise model, using only the default settings, run:

```
python make_test_train.py --example
python generate.py -model notewise_generator -output notewise_generation_samples
```

The output samples will be in data/output/notewise_generation_samples, or open Playlist.ipynb to listen to the output samples. I recommend the free program <a href="https://musescore.org/en">MuseScore</a> to translate the midi files into sheet music.

Note, you must first make sure the requirements (above) are installed.


<h2>Data:</h2>
If you use your own midi files, they should go in data/composers/midi/piano_solo or data/composers/midi/chamber (the project expects to see a folder of midi files for each composer, ie: data/composers/midi/piano_solo/bach/example_piece.mid). <br>

Run:

```
python midi-to-encoding.py
```

to translate midi files to text files in the various notewise and chordwise options. <br>
<br>
My dataset is available here (you can download any or all):<br>
Put these in data/composers/notewise:
<ul>
<li><a href="http://www.christinemcleavey.com/files/notewise_piano_solo.tar.gz">Notewise piano solo text files</a></li>
<li><a href="http://www.christinemcleavey.com/files/jazz.tar.gz">Notewise jazz piano solo text files</a></li>
<li><a href="http://www.christinemcleavey.com/files/notewise_chamber.tar.gz">Notewise piano/violin text files</a></li>
</ul>
Put these in data/composers/chordwise:
<ul>
<li><a href="http://www.christinemcleavey.com/files/chordwise_piano_solo.tar.gz">Chordwise piano solo text files</a></li>
<li><a href="http://www.christinemcleavey.com/files/jazz_chords.tar.gz">Chordwise jazz piano solo text files</a></li>  
<li><a href="http://www.christinemcleavey.com/files/chordwise_chamber.tar.gz">Chordwise piano/violin text files</a></li>
</ul>
(Run tar -zxvf thisfilename.tar.gz to expand each one.)


<h2>Training and Generation:</h2>
<ul>
<li>make_test_train.py - create the training and testing datasets (adjust notewise/chordwise, optionally create only a small sample size)</li>
<li>train.py - train an AWD-LSTM (adjust model parameters, dropout, and training regime)</li>
<li>generate.py - generate new samples (adjust generation size)</li>
</ul>
Each script has default settings which should be reasonable, but use --help to see the different options and parameters which can be modified.<br><br>
If you use the data files I've linked above, those are quite large, and will take a long time to train. If you are looking to experiment with different training networks, I'd highly recommend at first using --sample .1 (10% of the data) with make_test_train.py, so that you have a much smaller dataset to play with and can iterate faster.<br><br>
Playlist.ipynb is a simple Jupyter Notebook which creates a nicely formatted playlist for listening to all the generations.
<h2>Music Critic:</h2>
<ul>
  <li>make_critic_data.py - create the training and test datasests (requires a trained generation model to create the fake data)</li>
  <li>critic.py - trains a classifier to predict if a sample is human-composed or LSTM-composed
</ul>
<h2>Composer Classifier:</h2>
<ul>
  <li>make_composer_data.py - create the training and test datasests (all from human composed pieces)</li>
  <li>composer_classifier.py - trains a classifier to predict which human composed the piece
</ul>

<h2>Pretrained Models:</h2>
Sample pretrained models are included in this repository. They were trained using the default settings (all composers, notewise using a sample frequency 12, chordwise using a sample frequency 4). 
<ul>
  <li>notewise_generator</li>
  <li>chordwise_generator </li>
  <li>chamber_generator (uses notewise encoding)</li>
  <li>notewise_critic</li>
  <li>notewise_composer_classifier</li>
</ul>
  
For example, use:

```
python generator.py -model notewise_generator -output notewise_generation_samples --random_freq 0.8 --trunc 3
```

to generate musical samples.
