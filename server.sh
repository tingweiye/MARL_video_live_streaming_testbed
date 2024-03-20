cd server
eval "$(conda shell.bash hook)"
conda activate marl
gunicorn routing:application -c gunicorn_config.py
#python manage.py runserver 10.0.10.2:8080
