from django.http import FileResponse, Http404, HttpResponse, StreamingHttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.apps import apps
import asyncio
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
server = Server("HMARL")
shared_data_lock = threading.Lock()
shared_register_lock = threading.Lock()

async def async_file_iterator(file_content, chunk_size=8192):
    index = 0
    while index < len(file_content):
        chunk = file_content[index:index + chunk_size]
        yield chunk
        index += chunk_size
        await asyncio.sleep(0)

# Handle video download action
async def download_video(request, video_filename):
    print("   ")
    t0 = time.time()
    client_idx = int(request.META.get('HTTP_IDX'))
    request_gop = int(request.META.get('HTTP_GOP'))
    request_rate = float(request.META.get('HTTP_RATE'))
    estimated_bw = float(request.META.get('HTTP_BW'))
    client_buffer = float(request.META.get('HTTP_BUFFER'))
    client_freeze = float(request.META.get('HTTP_FREEZE'))
    client_latency = float(request.META.get('HTTP_LATENCY'))
    client_jump = float(request.META.get('HTTP_JUMP'))
    
    algo = request.META.get('HTTP_ALGO')
    
    # t1 = time.time()
    server_time = server.get_server_time()
    info = {"rate": request_rate,
            "bw": estimated_bw,
            "buffer": client_buffer,
            "freeze": client_freeze,
            "latency": client_latency,
            "jump": client_jump,
            "startTime":server_time}
    
    # update information of the client
    server.update_client(client_idx, info)
    
    # HMARL rate selection
    if algo == "HMARL":
        request_rate, intrinsic_reward, extrinsic_reward = server.hmarl_solve(client_idx)
        
    # print(server.check_pred(server_time))
    suggestion, video_filename, prepare = server.process_request(client_idx, request_gop, request_rate)
    
    video_path = os.path.join(os.getcwd(), "data/"+video_filename)
    # t2 = time.time()
    if os.path.exists(video_path):
        # lower, upper = server.encoder.check_range()
        Logger.log(f"Requested client: {client_idx}, gop: {request_gop}, rate: {request_rate}, server range: {server.encoder.check_range()}")
        
        with open(video_path, 'rb') as video_file:
            file_content = video_file.read()
            response = StreamingHttpResponse(async_file_iterator(file_content))
            
        response['Content-Type'] = 'video/mp4'
        response['Content-Disposition'] = f'attachment; filename="{video_filename}"'
        response['Server-Time'] = server_time
        response['Prepare-Time'] = prepare
        response['Suggestion'] = suggestion
        response['Rate'] = request_rate
        if algo == "MARL":
            instruction, fair_bw, reward = server.marl_solve(client_idx)
            response['Instruction'] = instruction
            response['Fairbw'] = fair_bw
            response['Reward'] = reward
        elif algo == "HMARL":
            response['Reward'] = intrinsic_reward
            response['ExReward'] = extrinsic_reward
        
        if request_gop + 1 != suggestion:
            Logger.log(f"Client {client_idx} latency too high, suggested downloading {suggestion - 1}")
        Logger.log(f"Client {client_idx} downloaded video segment {video_filename}")
        # t3 = time.time()
        # if client_idx == 1:
        #     print(f"process: {t2 - t1}, get file: {t3 - t2}, prepare: {prepare}, parsing: {t1 - t0}")
        return response
    else:
        raise Http404("Video not found")
    
# Handle client registry
@require_POST
@csrf_exempt
def client_register(request):
    with shared_register_lock:
        weight = float(request.META.get('HTTP_WEIGHT'))
        idx, suggestion = server.register_client(weight)
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

