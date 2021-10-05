# CornBot-BackEnd

Requirements:
  python 3.9.1

# Step to get up an running:
  1. Create a python virtual enviroment and name it venv
  2. `pip install -r requirements.txt`  -- the file is under
  3. CornBot-BackEnd/requirement.txt
  4. create tables and migrations
    python manage.py makemigrations
    python manage.py migrate
  5. create a superuser
    `python manage.py createsuperuser` **Note: follow the instructions to create your superuser after the command is ran**
  6. Run the server
   `python manage.py runserver 8000`
  7. You are all up and running!
  

  # Release Notes:
    ## 1.0.0
        1. User has the ability to register themselves in the system.
        2. User can be logged in and authenticated in the system.
        3. Basic Tutorial end-points to test authentication feature.
