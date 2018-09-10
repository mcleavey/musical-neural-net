
# Code taken directly from:
# https://stackoverflow.com/questions/33417151/playing-mp3-in-a-folder-with-jupyter-notebook
# (I modified it only to sort the playlist and to play only .wav files)

# Refer also to:
# http://devblog.lastrose.com/html5-audio-video-playlist/

import os

def make_playlist(music_dir):
    playlist_html=""
    audio_html=""
    count=0
    for root, dirs, files in os.walk(music_dir):
        for file in sorted(files):
            if file[-3:]!="wav":
                continue

            if count==0:
                playlist_html = u'''<li class="active"><a href="{0}">{1}</a>
                    </li>\n'''.format(os.path.join(root, file), file)
                audio_html = u'''<audio id="audio" preload="auto" tabindex="0" controls="" type="audio/mpeg">
                    <source type="audio/mp3" src="{}">Sorry, your browser does not support HTML5 audio.
                    </audio>'''.format(os.path.join(root, file))
            else:
                playlist_html +=u'''<li><a href="{0}">{1}</a></li>\n'''.format(os.path.join(root, file), file)
            count += 1

    playlist_html = audio_html + u'''\n<ol id="playlist">\n{}</ol>'''.format(playlist_html)
    playlist_css = """
    <style>
    #playlist .active a{color:#CC0000;text-decoration:none;}
    #playlist li a:hover{text-decoration:none;}
    </style>
    """
    
    playlist_js = """
    <script>
    var audio;
    var playlist;
    var tracks;
    var current;
    
    init();
    function init(){
        current = 0;
        audio = $('audio');
        playlist = $('#playlist');
        tracks = playlist.find('li a');
        len = tracks.length - 1;
        audio[0].volume = .90;
        playlist.find('a').click(function(e){
            e.preventDefault();
            link = $(this);
            current = link.parent().index();
                    run(link, audio[0]);
        });
        audio[0].addEventListener('ended',function(e){
            current++;
            if(current == len){
                current = 0;
                link = playlist.find('a')[0];
            }else{
                link = playlist.find('a')[current];    
            }
            run($(link),audio[0]);
        });
    }
    function run(link, player){
            player.src = link.attr('href');
            par = link.parent();
            par.addClass('active').siblings().removeClass('active');
            audio[0].load();
            audio[0].play();
    }
    </script>
    """
    return playlist_html, playlist_css, playlist_js
        
    
def make_http_playlist(music_dir):
    # Website Colors: ffe74c-ff5964-ffffff-38618c-35a7ff
    playlist_html=""
    audio_html=""
    head=""
    end_html=""
    image_html=""
    count=0
    for root, dirs, files in os.walk(music_dir):
        for file in sorted(files):
            if file[-3:]!="wav":
                continue
            if file[:4]=="prom":
                continue                
            if count==0:
                head=u'''<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
</head> <body><div class="jumbotron">
<h2 class="display-4" align="center">Clara: A Musical Neural Net Generator</h2>
<p align="center" class="lead">by Christine Payne</p>
</div>'''
                image_html=u'''<div id="composing" class="center" text-align="center"><h2>Composing new pieces, please wait...</h2><br><img src="./composing.gif"></div>'''
                audio_html = u'''<div id="view_playlist" class="center"><audio id="audio" preload="auto" tabindex="0" controls="" type="audio/wav">
                    <source type="audio/wav" src="{}">Sorry, your browser does not support HTML5 audio.
                    </audio> 
                      
                    '''.format(os.path.join(root, file))
                playlist_html = u'''<li class="active"><a href="{0}">{1}</a>
                    </li>\n'''.format(os.path.join(root, file), file)                
            else:
                playlist_html +=u'''<li><a href="{0}">{1}</a></li>\n'''.format(os.path.join(root, file), file)
            count += 1
    end_html=u'''<a class="btn btn-outline-secondary" href="./music" role="button" id="new">Generate New Songs (Takes
    ~1 Minute)</a>
    </div>



    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
  </body>
</html>'''
    playlist_html = head + image_html + audio_html + u'''\n<ul id="playlist">\n{}</ul>'''.format(playlist_html) + end_html
    playlist_css = """
    <style>
    hr {
    border-top: 3px double #38618c;
    width: 80%;
    }
    #playlist .active a{color:#FF5964;text-decoration:none;}
    #playlist li a:hover{text-decoration:none;color:#38618c}
    ul {
    columns: 3;
    -webkit-columns: 3;
    -moz-columns: 3;
     }
    img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 40%;
    }
    .center {
    margin: auto;
    width: 80%;
    border: 3px solid #35a7ff;
    padding: 10px;
    }
    </style>
    """
    
    playlist_js = """
    <script>
    var audio;
    var playlist;
    var next_button;
    var new_button;
    var view_playlist;
    var composing_image;
    var tracks;
    var current;
    
    init();
    function init(){
        current = 0;
        audio=$('audio');
        playlist = $('#playlist');
        next_button = $('#next');
        new_button = $('#new');
        view_playlist = $('#view_playlist');
        composing_image = $('#composing');
        composing_image.hide();
        view_playlist.show();
        tracks = playlist.find('li a');
        len = tracks.length - 1;
        audio[0].volume = .99;
        playlist.find('a').click(function(e){
            e.preventDefault();
            link = $(this);
            current = link.parent().index();
                    run(link, audio[0]);
        });     
        next_button.click(function(){
            console.log("Clicked button")
            current++;
            if(current == len){
                current = 0;
                link = playlist.find('a')[0];
            }else{
                link = playlist.find('a')[current];    
            }
            run($(link),audio[0]);
        });
        new_button.click(function(){
            console.log("Create new generations")
            view_playlist.hide();
            composing_image.show();
        });
        audio[0].addEventListener('ended',function(e){
            current++;
            if(current == len){
                current = 0;
                link = playlist.find('a')[0];
            }else{
                link = playlist.find('a')[current];    
            }
            run($(link),audio[0]);
        });
    }
    function run(link, player){
            player.src = link.attr('href');
            par = link.parent();
            par.addClass('active').siblings().removeClass('active');
            audio[0].load();
            audio[0].play();
    }
    </script>
    """
    return playlist_html, playlist_css, playlist_js
        
