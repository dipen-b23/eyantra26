from setuptools import find_packages, setup

package_name = 'task_1a'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eYantra',
    maintainer_email='user@example.com',
    description='Task 1A - Camera Detection',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_detection = task_1a.camera_detection:main',
        ],
    },
)
