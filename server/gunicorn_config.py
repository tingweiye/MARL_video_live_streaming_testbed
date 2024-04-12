import os

GUNICORN_SERVER = None

def on_starting(server):
    # print("hello")
    # global GUNICORN_SERVER
    # GUNICORN_SERVER = server
    # GUNICORN_SERVER.trail = os.getenv('TRAIL', 0)
    pass

bind = "127.0.0.1:8080"
workers = 1
timeout = 120
keepalive = 120
worker_class = 'uvicorn.workers.UvicornWorker'
app_module = 'server'
app_callable = 'videoServer'