from setuptools import setup, find_packages

setup(
    name='detection_pipeline_ros',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'opencv-python>=4.5.0',
        'ultralytics>=8.0.0',
        'supervision>=0.3.0',
        'numpy>=1.21.0',
        'PyYAML>=5.4',
    ],
    author='Baraa',
    author_email='baraa@example.com',
    description='UAV branch detection pipeline for autonomous landing',
    license='MIT',
)
