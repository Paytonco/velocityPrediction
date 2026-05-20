import os
import sys
import pathlib


def initialize_django_in_notebook(settings_module='project.settings', PROJECT_DIR=pathlib.Path(__file__).resolve().parent.parent):
    # Add the project root to the sys.path so you can import your Django app modules
    sys.path.append(str(PROJECT_DIR))

    # Set your settings module and allow async operations
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)
    os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

    import django
    django.setup()
