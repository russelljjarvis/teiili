import os
from pathlib import Path

from teili.models import neuron_models, synapse_models

def generate_user_directory():
    path = os.path.expanduser("~")
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    teili_home = Path(current_file_path).parent.parent

    equation_path = os.path.join(path, "teiliApps", "equations")

    neuron_models.main(path=equation_path)
    synapse_models.main(path=equation_path)

    source_path = os.path.join(teili_home, "tests", "")
    target_path = os.path.join(path, "teiliApps", "unit_tests", "")

    if not os.path.isdir(target_path):
        Path(target_path).mkdir(parents=True)

    os.system('cp -r {}* {}'.format(source_path, target_path))

    source_path = os.path.join(teili_home, "tutorials", "")
    target_path = os.path.join(path, "teiliApps", "tutorials", "")
    if not os.path.isdir(target_path):
        Path(target_path).mkdir(parents=True)

    os.system('cp -r {}* {}'.format(source_path, target_path))

if __name__ == '__main__':
    generate_user_directory()
