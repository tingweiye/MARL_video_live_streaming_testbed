class Config:
    
    # Basics
    SEG_DURATION = 1.0
    CHUNK_DURATION = 0.2
    CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
    BITRATE = [0.3, 0.5, 1.0, 2.0, 3.0, 6.0]
    SPEED = [0.9, 1.0, 1.1]
    MAX_RATE = BITRATE[-1]
    FPS = 24
    
    # clients
    INITIAL_DUMMY_LATENCY = 3.0
    INITIAL_RATE = 3.0
    INITIAL_LATENCY = 1.5
    CLIENT_MAX_BUFFER_LEN = 5
    MAX_HISTORY = 10000
    
    # server
    ENCODING_TIME = 1.0
    PUSEDO_ENCODE_TIME = ENCODING_TIME
    SERVER_MAX_BUFFER_LEN = 5
    SEG_NUM = 10
    
    VIDEO_FORMAT = ".mp4"