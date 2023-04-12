import os


# def safe_import(package_name: str, module_name: str = None):
#     try:
#         if module_name is not None:
#             import_name = package_name + '.' + module_name
#         else:
#             import_name = package_name
#         module = __import__(import_name, fromlist=[module_name])
#     except ImportError as e:
#         print(f"{e}: Import {package_name} failed, try to install")
#         import subprocess
#         subprocess.call(['pip', 'install', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', package_name])
#         module = __import__(import_name, fromlist=[module_name])
#     return module


def install_package(package_name: str, pip_version=2, python_path: str = None):
    if python_path is None:
        python_path = 'https://pypi.tuna.tsinghua.edu.cn/simple'
    if pip_version == 2:
        pip_version_str = ''
    elif pip_version == 3:
        pip_version_str = '3'
    else:
        raise ValueError(f"Invalid pip version: {pip_version}, must in {2, 3}")
    import_info = 'pip' + pip_version_str + ' install ' + package_name + ' -i ' + python_path
    os.system(import_info)
