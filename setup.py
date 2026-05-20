from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["detection_pipeline_ros", "detection_pipeline_ros.msg"],
    package_dir={"": "."},
)

setup(**d)
