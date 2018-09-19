from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import argparse


USE_HTTPS = False
from playlist import make_http_playlist
from generate import *

music_dir = u'./data/output/temp'

model="mod"
training="light"
random_freq=.8
trunc=2


class MyHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.startswith('/kill_server'):
            print("Server is going down, run it again manually!")
            def kill_me_please(server):
                server.shutdown()
            threading.start_new_thread(kill_me_please, (httpd,))
            self.send_error(500)
    # GET
    def do_GET(self):
        # Send response status code
        if self.path.endswith('wav') or self.path.endswith('gif'):
            super(MyHandler, self).do_GET()
            return
        if self.path.endswith('music'):
            self.send_response(200)
            self.send_header('Location','/')
            self.send_header('Content-type','text/html')
            self.end_headers()  
            
            main(model, training, "test", "train", 600, 4, False, 33, False, "temp", 16, trunc, random_freq, 240)            
            playlist_html, playlist_css, playlist_js=make_http_playlist(music_dir)
            self.wfile.write(bytes(playlist_html, "utf8"))
            self.wfile.write(bytes(playlist_css, "utf8"))
            self.wfile.write(bytes(playlist_js, "utf8"))          
            
            return        

        playlist_html, playlist_css, playlist_js=make_http_playlist(music_dir)
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        self.wfile.write(bytes(playlist_html, "utf8"))
        self.wfile.write(bytes(playlist_css, "utf8"))
        self.wfile.write(bytes(playlist_js, "utf8"))

        return
    
    



class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass

def run():
    server_address = ('', 8000)
    server = ThreadingSimpleServer(server_address, MyHandler)
    if USE_HTTPS:
        import ssl
        server.socket = ssl.wrap_socket(server.socket, keyfile='./key.pem', certfile='./cert.pem', server_side=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Generative Model")
    parser.set_defaults(model="mod")
    parser.add_argument("--training", dest="training", help="Training (default light)")
    parser.set_defaults(training="light")
    parser.add_argument("--random_freq", dest="random_freq", help="Frequency of randomized choice (default .8)")
    parser.set_defaults(random_freq=.8)
    parser.add_argument("--trunc", dest="trunc", help="Number of top predictions to consider (default 2)")
    parser.set_defaults(trunc=2)

    args = parser.parse_args()
    model,training,random_freq,trunc=args.model,args.training,args.random_freq,args.trunc                    
    run()