import os
  
GUNICORN_SERVER = None

def on_starting(server):
    pass

bind = "127.0.0.1:8080"
workers = 1
# threads = 3
timeout = 120
keepalive = 120
worker_class = 'uvicorn.workers.UvicornWorker'
# worker_class = 'gunicorn.workers.ggevent.GeventWorker'
app_module = 'server'
app_callable = 'videoServer'