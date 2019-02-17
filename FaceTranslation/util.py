import os


project_root = os.path.abspath(__file__)
for _ in range(2):
    project_root = os.path.dirname(project_root)
