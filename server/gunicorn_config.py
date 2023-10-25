# gunicorn_config.py

# from videoServer.server import Server

# server = Server()

def on_starting(server):
    # 这里可以添加其他启动时的操作
    pass

bind = "127.0.0.1:8080"
workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'
app_module = 'server'
app_callable = 'videoServer'