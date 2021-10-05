# CornBot-BackEnd

Requirements:
  python 3.9.1

Step to get up an running:
  Create a python virtual enviroment and name it venv
  pip install -r requirements.txt  -- the file is under CornBot-BackEnd/requirement.txt
  create tables and migrations
    python manage.py makemigrations
    python manage.py migrate
  create a superuser
    python manage.py createsuperuser Note: follow the instructions to create your superuser after the command is ran
  Run the server
    python manage.py runserver 8000
  You are all up and running!
  
