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
        