from setuptools import setup
import os
from glob import glob

package_name = 'potential_fields_lab'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'models/obstacle_box'),
            glob('models/obstacle_box/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'potential_field_1d = potential_fields_lab.potential_field_1d:main',
        ],
    },
)
