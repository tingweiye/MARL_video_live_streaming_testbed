from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
import threading
import os
import time
import sys
# from gunicorn_config import server
sys.path.append("..")
from videoServer.server import Server
from utils.utils import Logger
from utils.config import Config

# VIDEO_FORMAT = ".mp4"

# Server initialization
# server = apps.get_app_config('videoServer').server
server = Server()
shared_data_lock = threading.Lock()
shared_register_lock = threading.Lock()

# Handle video download action
def download_video(request, video_filename):
    print("   ")
    client_idx = int(request.META.get('HTTP_IDX'))
    request_gop = int(request.META.get('HTTP_GOP'))
    request_rate = float(request.META.get('HTTP_RATE'))
    print(f"Requested client: {client_idx}, gop: {request_gop}, rate: {request_rate}, server range: {server.encoder.check_range()}")
    
    t1 = time.time()
    suggestion, video_filename, prepare = server.process_request(request_gop, request_rate)
    video_path = os.path.join(os.getcwd(), "data/"+video_filename)
    # print(f"request: {request_gop}, range: {server.check_range()}")
    t2 = time.time()
    if os.path.exists(video_path):
        lower, upper = server.encoder.check_range()
        print(lower, upper, server.get_server_time())
        video_file = open(video_path, 'rb')
        response = FileResponse(video_file)
        response['Content-Type'] = 'video/mp4'
        response['Content-Disposition'] = f'attachment; filename="{video_filename}"'
        response['Server-Time'] = server.get_server_time()
        response['Prepare-Time'] = prepare
        response['suggestion'] = suggestion
        if request_gop + 1 != suggestion:
            Logger.log(f"Client {client_idx} latency too high, suggested downloading {suggestion - 1}")
        Logger.log(f"Client {client_idx} downloaded video segment {video_filename}")
        t3 = time.time()
        print(f"process: {t2 - t1}, get file: {t3 - t2}")
        return response
    else:
        raise Http404("Video not found")
    
# Handle client registry
@require_POST
@csrf_exempt
def client_register(request):
    with shared_register_lock:
        idx, suggestion = server.register_client()
        Logger.log(f"Client {idx} successfully connected to the server")
    
    response = HttpResponse(f"Client {idx} registered")
    response['idx'] = idx
    response['next'] = suggestion
    return response

# Handle client exits
@require_POST
@csrf_exempt
def client_exit(request):
    with shared_register_lock:
        idx = int(request.META.get('HTTP_IDX'))
        server.client_exit(idx)
        Logger.log(f"Client {idx} successfully exited from the server")
        
    response = HttpResponse(f"Client {idx} registered")
    return response

