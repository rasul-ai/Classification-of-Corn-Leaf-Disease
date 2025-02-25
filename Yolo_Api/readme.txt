terminal command:
flask --app app_html.py run --host=0.0.0.0

# Build image
docker build -t corn_leaf_disease_classifier .

# Now run docker container
docker run -p 5000:5000 corn_leaf_disease_classifier