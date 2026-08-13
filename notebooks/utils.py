import os
import sys
import pathlib


DIR_ROOT = pathlib.Path(__file__).resolve().parent.parent


def initialize_django_in_notebook(settings_module='project.settings', project_dir=DIR_ROOT):
    # Add the project root to the sys.path so you can import your Django app modules
    sys.path.append(str(project_dir))

    # Set your settings module and allow async operations
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)
    os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'true')

    import django
    django.setup()
