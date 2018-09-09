from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

USE_HTTPS = False
from playlist import make_http_playlist
from generate import *

music_dir = u'./data/output/temp'

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
            
            main("notewise4b", "full", "test", "train", 600, 4, False, 33, False, "temp", 16, 3, 1, 200)            
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
    run()