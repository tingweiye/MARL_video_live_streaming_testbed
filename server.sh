cd server
eval "$(conda shell.bash hook)"
conda activate marl
# Initialize counter
counter=0

while [ $counter -lt 50 ]; do
    # Execute the Python script with counter as parameter
    echo $counter > .control
    gunicorn routing:application -c gunicorn_config.py
    # Increment counter
    counter=$((counter+1))
done

# gunicorn routing:application -c gunicorn_config.py

