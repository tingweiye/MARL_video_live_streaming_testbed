from django.apps import AppConfig
from videoServer.server import Server


class VideoserverConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'videoServer'
    
    def ready(self):
        # print("APP readying...")
        self.server = Server()
