# musical-neural-net
Train an <a href="https://arxiv.org/pdf/1708.02182.pdf">AWD-LSTM</a> to generate piano or violin/piano music<br>
Project structure overview is <a href="http://christinemcleavey.com/music-generator-project-structure/">here</a>.<br>
Sample generations are <a href="http://christinemcleavey.com/human-or-ai/">here</a>.<br>
Detailed paper is **TO DO** 

<h2>Requirements:</h2>
<ul>
<li>PyTorch version 0.3.0.post4</li>
<li><a href="https://github.com/fastai/fastai">FastAI</a>: install link in musical-neural-net home directory</li>
<li><a href="http://schristiancollins.com/generaluser.php">GeneralUser</a>: install in data folder (this is needed to translate midi files to mp3)</li>
</ul>

<h2>Data:</h2>
If you use your own midi files, they should go in data/composers/midi/piano_solo or data/composers/midi/chamber (the project expects to see a folder of midi files for each composer, ie: data/composers/midi/piano_solo/bach/example_piece.mid). <br>

Run:

```
python midi-to-encoding.py
```

to translate midi files to text files in the various notewise and chordwise options. <br>
My dataset is available here:<br>
Put these in data/composers/midi/piano_solo<br>
<a href="http://www.christinemcleavey.com/files/notewise.tar.gz">Notewise piano solo text files</a><br>
<a href="http://www.christinemcleavey.com/files/chordwise.tar.gz">Chordwise piano solo text files</a><br>
Put these in data/composers/midi/chamber<br>
<a href="http://www.christinemcleavey.com/files/notewise.tar.gz">Notewise piano/violin text files</a><br>
<a href="http://www.christinemcleavey.com/files/chordwise.tar.gz">Chordwise piano/violin text files</a><br>

(Run ` tar -zxvf <name>.tar.gz ` to expand each one.)


<h2>Training and Generation:</h2>
<ul>
<li>make_test_train.py - create the training and testing datasets (adjust notewise/chordwise, optionally create only a small sample size)</li>
<li>train.py - train an AWD-LSTM (adjust model parameters, dropout, and training regime)</li>
<li>generate.py - generate new samples (adjust generation size)</li>
</ul>
Each script has default settings which should be reasonable, but use --help to see the different options and parameters which can be modified.<br>
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
