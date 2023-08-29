from setuptools import find_packages, setup

package_name = 'epsrc_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
        'epsrc_controller',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['launch/epsrc_controller.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joseph Agrane',
    maintainer_email='josephagrane@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = epsrc_controller.controller:main',
            'communicator = epsrc_controller.communicator:main'
        ],
    },
)