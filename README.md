# CornBot-BackEnd

Requirements:
  python 3.9.1

# Steps to get up and running:
  1. Create a python virtual enviroment and name it venv
  2. `pip install -r requirements.txt`  -- the file is under
   CornBot-BackEnd/requirement.txt
  3. create tables and migrations
    python manage.py makemigrations
    python manage.py migrate
  4. create a superuser
    `python manage.py createsuperuser` **Note: follow the instructions to create your superuser after the command is ran**
  5. Run the server
   `python manage.py runserver 8000`
  6. You are all up and running!
  

  # Release Notes:
    ## 1.0.0 Milestone 1
        1. User has the ability to register themselves in the system.
        2. User can be logged in and authenticated in the system.
        3. Basic Tutorial end-points to test authentication feature.
  
    ## 1.1.0 Milestone 2
        1. Database can be populated with images from AWS S3.
        2. Backend can be used iwth SQLite or MySQL databases. 
        3. Endpoints for front end to fetch images and post user labels. 
        4. All necessary models and serializers created. 
    ## 1.2.0 Milestone 3
        1. Choices gained date time field for creation date, boolean for if users choice is for user training or not.
        2. New endpoint to return users accuracies for front-end leaderboard.
        3. Documentation added for views and models.
        4. Code cleanup: removed not needed functions, rename view methods.
    ## 1.3.0 Milestone 4
        1. Added new endpoint getUpload. It spots all blights using bounding box in a image, uploaded by user.
        2. Added a analytics endpoint that shows user labeling accuracy.
        3. Added second analytics endpoint that shows user testing accuracy.
        4. Added thrid analytics endpoint that shows images that has been most frequently misclassified. 
