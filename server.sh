cd server
eval "$(conda shell.bash hook)"
conda activate marl
python manage.py runserver 8080
