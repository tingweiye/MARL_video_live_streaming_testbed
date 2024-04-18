from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application()   
    }
)
# from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.auth import AuthMiddlewareStack
# from server import urls

# application = ProtocolTypeRouter({
#     "http": AuthMiddlewareStack(
#         URLRouter(
#             urls.urlpatterns
#         )
#     ),
# })
